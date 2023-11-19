import asyncio
import json
import logging
import traceback
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    HTMLResponse,
    FileResponse,
)

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
)
from compel import Compel
import torch

from canny_gpu import SobelOperator

try:
    import intel_extension_for_pytorch as ipex
except:
    pass
from PIL import Image
import numpy as np
import gradio as gr
import io
import uuid
import os
import time
import psutil


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 2))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "False")
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)

WIDTH = 512
HEIGHT = 512

MODEL_ID =  os.environ.get("MODEL_ID", "wavymulder/Analog-Diffusion") 
LCM_LORA_ID =  os.environ.get("LCM_LORA_ID", "latent-consistency/lcm-lora-sdv1-5") 
CONTROLNET_MODEL_ID=os.environ.get("CONTROLNET_MODEL_ID", "lllyasviel/control_v11p_sd15_seg") 

# check if MPS is available OSX only M1/M2/M3 chips
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "xpu" if xpu_available else "cpu"
)

# change to torch.float16 to save GPU memory
torch_dtype = torch.float16

print(f"TIMEOUT: {TIMEOUT}")
print(f"SAFETY_CHECKER: {SAFETY_CHECKER}",SAFETY_CHECKER == "True")
print(f"MAX_QUEUE_SIZE: {MAX_QUEUE_SIZE}")
print(f"device: {device}")

if mps_available:
    device = torch.device("mps")
    device = "cpu"
    torch_dtype = torch.float32

controlnet_model = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_ID, torch_dtype=torch_dtype
).to(device)

canny_torch = SobelOperator(device=device)

models_id = [
    MODEL_ID,
]
lcm_lora_id = LCM_LORA_ID

if SAFETY_CHECKER == "True":
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            controlnet=controlnet_model,
        )
else:
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            controlnet=controlnet_model,
        )

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
pipe.to(device=device, dtype=torch_dtype).to(device)

if psutil.virtual_memory().total < 64 * 1024**3:
    pipe.enable_attention_slicing()

# Load LCM LoRA
pipe.load_lora_weights(lcm_lora_id, adapter_name="lcm")

compel_proc = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    truncate_long_prompts=False,
)
if TORCH_COMPILE:
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

    pipe(
        prompt="warmup",
        image=[Image.new("RGB", (768, 768))],
        control_image=[Image.new("RGB", (768, 768))],
    )


user_queue_map = {}


class InputParams(BaseModel):
    seed: int = 2159232
    prompt: str
    guidance_scale: float = 8.0
    strength: float = 0.5
    steps: int = 4
    lcm_steps: int = 50
    width: int = WIDTH
    height: int = HEIGHT
    controlnet_scale: float = 0.8
    controlnet_start: float = 0.0
    controlnet_end: float = 1.0
    canny_low_threshold: float = 0.31
    canny_high_threshold: float = 0.78
    debug_canny: bool = False
    model_id: str = MODEL_ID


def predict(input_image: Image.Image, params: InputParams):
    generator = torch.manual_seed(params.seed)

    control_image = canny_torch(
        input_image, params.canny_low_threshold, params.canny_high_threshold
    )
    prompt_embeds = compel_proc(params.prompt)
    # pipe = pipes[params.model_id]
    results = pipe(
        control_image=control_image,
        prompt_embeds=prompt_embeds,
        generator=generator,
        image=input_image,
        strength=params.strength,
        num_inference_steps=params.steps,
        guidance_scale=params.guidance_scale,
        width=params.width,
        height=params.height,
        output_type="pil",
        controlnet_conditioning_scale=params.controlnet_scale,
        control_guidance_start=params.controlnet_start,
        control_guidance_end=params.controlnet_end,
    )
    nsfw_content_detected = (
        results.nsfw_content_detected[0]
        if "nsfw_content_detected" in results
        else False
    )
    if nsfw_content_detected:
        return None
    result_image = results.images[0]
    if params.debug_canny:
        # paste control_image on top of result_image
        w0, h0 = (200, 200)
        control_image = control_image.resize((w0, h0))
        w1, h1 = result_image.size
        result_image.paste(control_image, (w1 - w0, h1 - h0))

    return result_image


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if MAX_QUEUE_SIZE > 0 and len(user_queue_map) >= MAX_QUEUE_SIZE:
        print("Server is full")
        await websocket.send_json({"status": "error", "message": "Server is full"})
        await websocket.close()
        return

    try:
        uid = str(uuid.uuid4())
        print(f"New user connected: {uid}")
        await websocket.send_json(
            {"status": "success", "message": "Connected", "userId": uid}
        )
        user_queue_map[uid] = {"queue": asyncio.Queue()}
        await websocket.send_json(
            {"status": "start", "message": "Start Streaming", "userId": uid}
        )
        await handle_websocket_data(websocket, uid)
    except WebSocketDisconnect as e:
        logging.error(f"WebSocket Error: {e}, {uid}")
        traceback.print_exc()
    finally:
        print(f"User disconnected: {uid}")
        queue_value = user_queue_map.pop(uid, None)
        queue = queue_value.get("queue", None)
        if queue:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue


@app.get("/queue_size")
async def get_queue_size():
    queue_size = len(user_queue_map)
    return JSONResponse({"queue_size": queue_size})


@app.get("/stream/{user_id}")
async def stream(user_id: uuid.UUID):
    uid = str(user_id)
    try:
        user_queue = user_queue_map[uid]
        queue = user_queue["queue"]

        async def generate():
            last_prompt: str = None
            while True:
                data = await queue.get()
                input_image = data["image"]
                params = data["params"]
                if input_image is None:
                    continue

                image = predict(
                    input_image,
                    params,
                )
                if image is None:
                    continue
                frame_data = io.BytesIO()
                image.save(frame_data, format="JPEG")
                frame_data = frame_data.getvalue()
                if frame_data is not None and len(frame_data) > 0:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"

                await asyncio.sleep(1.0 / 120.0)

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace;boundary=frame"
        )
    except Exception as e:
        logging.error(f"Streaming Error: {e}, {user_queue_map}")
        traceback.print_exc()
        return HTTPException(status_code=404, detail="User not found")


async def handle_websocket_data(websocket: WebSocket, user_id: uuid.UUID):
    uid = str(user_id)
    user_queue = user_queue_map[uid]
    queue = user_queue["queue"]
    if not queue:
        return HTTPException(status_code=404, detail="User not found")
    last_time = time.time()
    try:
        while True:
            data = await websocket.receive_bytes()
            params = await websocket.receive_json()
            params = InputParams(**params)
            pil_image = Image.open(io.BytesIO(data))

            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
            await queue.put({"image": pil_image, "params": params})
            if TIMEOUT > 0 and time.time() - last_time > TIMEOUT:
                await websocket.send_json(
                    {
                        "status": "timeout",
                        "message": "Your session has ended",
                        "userId": uid,
                    }
                )
                await websocket.close()
                return

    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("./static/controlnetlora.html")




def run_web_service():
    import sys
    import uvicorn
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='命令行参数示例')

    # 添加命令行参数
    parser.add_argument('--TIMEOUT', type=float, default=0, help='超时时间')
    parser.add_argument('--SAFETY_CHECKER', type=bool, default=False, help='安全检查器')
    parser.add_argument('--MAX_QUEUE_SIZE', type=int, default=2, help='最大队列大小')
    # parser.add_argument('uvicorn', nargs='+', help='uvicorn命令')

    parser.add_argument('--MODEL_ID',  type=str, default="wavymulder/Analog-Diffusion", help='基础模型')

    parser.add_argument('--LCM_LORA_ID',  type=str, default="latent-consistency/lcm-lora-sdv1-5", help='LCM-Lora模型')

    parser.add_argument('--CONTROLNET_MODEL_ID',  type=str, default="lllyasviel/control_v11p_sd15_seg", help='controlnet模型')

    parser.add_argument('--PORT', type=float, default=7860, help='端口')

    # 解析命令行参数
    args = parser.parse_args()

    # 获取命令行参数的值
    timeout = str(args.TIMEOUT)
    safety_checker = str(args.SAFETY_CHECKER)
    max_queue_size = str(args.MAX_QUEUE_SIZE)
    # uvicorn_command = args.uvicorn
    model_id = args.MODEL_ID
    lcm_lora_id=args.LCM_LORA_ID
    port=args.PORT

    # 设置环境变量
    os.environ['TIMEOUT'] = timeout
    os.environ['SAFETY_CHECKER'] = safety_checker
    os.environ['MAX_QUEUE_SIZE'] = max_queue_size
    os.environ['MODEL_ID'] = model_id
    os.environ['LCM_LORA_ID'] = lcm_lora_id
    os.environ['CONTROLNET_MODEL_ID'] = CONTROLNET_MODEL_ID

    
    uvicorn.run(app, host="127.0.0.1", port=port)

if __name__ == "__main__":
    run_web_service()
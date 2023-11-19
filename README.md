# 实时潜在一致性模型

这个演示展示了使用潜在一致性模型（LCM）和MJPEG流服务器的Diffusers进行图像到图像的控制。您需要一个摄像头来运行此演示。🤗

在这里可以看到一些实时演示的集合。

## 本地运行

您需要CUDA和Python 3.10，Mac配备M1/M2/M3芯片或Intel Arc GPU

TIMEOUT: 限制用户会话超时
SAFETY_CHECKER: 如果您想关闭NSFW过滤器，则禁用
MAX_QUEUE_SIZE: 限制当前应用实例的用户数量
TORCH_COMPILE: 如果您想使用torch compile进行更快的推理，则启用，适用于A100 GPU

### 安装

```bash
create_env.bat
```

### 图像到图像

```bash
uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload
```

### 图像到图像controlnet

Canny 算法流程来自 taabata

```bash
uvicorn "app-controlnet:app" --host 0.0.0.0 --port 7860 --reload
```

### 文本到图像

```bash
uvicorn "app-txt2img:app" --host 0.0.0.0 --port 7860 --reload
```

### LCM + LoRa

使用LCM-LoRa，使其能够在最多4个步骤中进行推理。

### 图像到图像controlnet Canny LoRa

```bash
uvicorn "app-controlnetlora:app" --host 0.0.0.0 --port 7860 --reload
```

### 文本到图像

```bash
uvicorn "app-txt2imglora:app" --host 0.0.0.0 --port 7860 --reload
```

### 设置环境变量

```bash
TIMEOUT=120 SAFETY_CHECKER=True MAX_QUEUE_SIZE=4 uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload
```

如果您在本地运行并希望在Mobile Safari上进行测试，则Web服务器需要通过HTTPS提供服务。

```bash
openssl req -newkey rsa:4096 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem
uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload --log-level info --ssl-certfile=certificate.pem --ssl-keyfile=key.pem
```

## Docker

您需要NVIDIA Container Toolkit for Docker

```bash
docker build -t lcm-live .
docker run -ti -p 7860:7860 --gpus all lcm-live
```

或者使用环境变量

```bash
docker run -ti -e TIMEOUT=0 -e SAFETY_CHECKER=False -p 7860:7860 --gpus all lcm-live
```

# 打包成APP
pip install -U huggingface_hub hf_transfer
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download --resume-download wavymulder/Analog-Diffusion --local-dir wavymulder/Analog-Diffusion

huggingface-cli download --resume-download latent-consistency/lcm-lora-sdv1-5 --local-dir latent-consistency/lcm-lora-sdv1-5

huggingface-cli download --resume-download lllyasviel/control_v11p_sd15_canny --local-dir lllyasviel/control_v11p_sd15_canny


venv/Scripts/python -s -m pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple

pyinstaller app-txt2imglora.py --clean
pyinstaller -F app-txt2imglora.py --clean
@echo off
python -m venv venv
echo Activating virtual environment...
venv\Scripts\python -s -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
venv\Scripts\activate
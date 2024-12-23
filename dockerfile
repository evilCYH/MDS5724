# base image
FROM python:3.12.3

# copy the requirements.txt into the image
COPY requirements.txt requirements.txt

# Install packages
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

# copy script into image
ADD app /app

# set working directory
WORKDIR /app

# run script when container starts
ENTRYPOINT ["python3", "main.py"]
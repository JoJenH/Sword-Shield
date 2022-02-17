# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


COPY ./ /opt/Sword_Shield
WORKDIR /opt/Sword_Shield

RUN python -m pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN echo "deb http://mirrors.ustc.edu.cn/debian bullseye main\ndeb http://mirrors.ustc.edu.cn/debian bullseye-updates main" > /etc/apt/sources.list
RUN apt-get update &&   apt-get -y install libnss3 xvfb gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2   libdbus-1-3 libexpat1 libfontconfig1 libgbm1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0   libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1   libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1   libxtst6 ca-certificates fonts-liberation libnss3 lsb-release xdg-utils wget vim libgomp1

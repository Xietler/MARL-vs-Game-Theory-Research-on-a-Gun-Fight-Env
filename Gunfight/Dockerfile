FROM python:3.6

RUN apt-get -y update && apt-get -y install ffmpeg
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/GridBattle


COPY . $CODE_DIR/algorithms
WORKDIR $CODE_DIR/algorithms

# 建议安装pytorch0.4.0避免遇到Pytorch1.0的坑
# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install pytorch 0.4.0 && \


CMD /bin/bash

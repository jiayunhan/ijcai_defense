# Dockerfile of Example
# Version 1.0
# Base Images
FROM registry.us-west-1.aliyuncs.com/bdu-xlab/torch_base:latest
#MAINTAINER
#FROM nvcr.io/nvidia/tensorflow:18.12-py3
MAINTAINER jiayunhan008

RUN mkdir /competition
WORKDIR /competition

RUN mkdir models
RUN mkdir utils
ADD models/w* ./models/
ADD utils/* ./utils/
ADD . /competition



# RUN mkdir ./models
#RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/inception_v1.tar.gz' && tar -xvf inception_v1.tar.gz -C ./models/
#RUN wget -P ./models/ https://github.com/jiayunhan/ijcai_defense/releases/download/release-v0.0.1/weights.pth.tar

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common nano wget git curl python3-dev python3-pip libsm6 libxext6
 
# clear cache
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install stable-baselines3[extra]==1.6.0

WORKDIR /workspace

COPY setup.py .

RUN pip3 install -e . && rm setup.py


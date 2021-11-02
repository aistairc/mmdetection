ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
ARG TIMEZONE="Asia/Tokyo"  # specify your region

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TZ=${TIMEZONE}

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 emacs python3-opencv tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/cocodataset/cocoapi/issues/172
RUN pip install cython

# Install MMCV
RUN pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
RUN cd /cocoapi/PythonAPI && make

# Install MMDetection
RUN conda clean --all
#RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone https://github.com/ishitsuka-hikaru/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/jupyter.txt
RUN pip install --no-cache-dir -e .

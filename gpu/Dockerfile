# Ubuntu 16.04, CUDA 8.0

FROM nvidia/cuda:8.0-runtime-ubuntu16.04

LABEL maintainer "MICROSOFT CORPORATION"

LABEL com.microsoft.cntk.version="2.0"

ENV CNTK_VERSION="2.0"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget && \

    apt-get -y autoremove && \

    rm -rf /var/lib/apt/lists/*


# Get CNTK Binary Distribution

RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \

    CNTK_SHA256="dab691a81602efe7c2c3729bff86e5934489c9f72cfc6911ae18bc909adf83b4" && \

    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict - && \

    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker

###### Get frcnn git repo and set up

RUN apt-get update && apt-get -y install git cmake libboost-all-dev

RUN git clone --depth 1 https://github.com/ruonanl/ObjectDetectionUsingCntk /fscnn

RUN bash -c "source /cntk/activate-cntk; pip install scikit-learn Pillow future EasyDict opencv-python"

###### prepare fscnn-specifics resources
	
RUN mkdir -p /fscnn/data/drone/testImages && \

    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/sampleimage/1000093223.jpg -P /fscnn/data/drone/testImages && \
	
    mkdir -p /fscnn/proc1/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/1/frcn_nn.model -P /fscnn/proc1/drone/models && \
	
    mkdir -p /fscnn/proc2/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/2/frcn_nn.model -P /fscnn/proc2/drone/models

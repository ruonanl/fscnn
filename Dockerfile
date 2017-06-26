# Ubuntu 14.04, CUDA 8.0

FROM nvidia/cuda:8.0-runtime-ubuntu14.04

LABEL maintainer "MICROSOFT CORPORATION"

LABEL com.microsoft.cntk.version="2.0"

ENV CNTK_VERSION="2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \

    # General

        ca-certificates \

        wget \

        && \

    # Clean-up

    apt-get -y autoremove \

        && \

    rm -rf /var/lib/apt/lists/*



# Get CNTK Binary Distribution

RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \

    CNTK_SHA256="dab691a81602efe7c2c3729bff86e5934489c9f72cfc6911ae18bc909adf83b4" && \

    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict - && \

    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \

    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker

# Get frcnn git repo and set up

RUN apt-get install git-all && \

    git clone https://github.com/Azure/ObjectDetectionUsingCntk /fscnn && \
	
    pip install -r /fscnn/resources/python35_64bit_requirements/requirements.txt
	
# prepare fscnn-specifics
	
RUN mkdir /fscnn/data/drone/testImages && \

    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/sampleimage/1000093223.jpg -P /fscnn/data/drone/testImages && \
	
    mkdir /fscnn/proc1/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/1/frcn_nn.model -P /fscnn/proc1/drone/models && \
	
    mkdir /fscnn/proc2/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/2/frcn_nn.model -P /fscnn/proc2/drone/models && \

    wget https://raw.githubusercontent.com/ruonanl/fscnn/master/6_scoreImage_together.py -P /fscnn && \
	
    wget https://raw.githubusercontent.com/ruonanl/fscnn/master/PARAMETERSall.py -P /fscnn && \
	
    wget https://raw.githubusercontent.com/ruonanl/fscnn/master/SLIDINGWINDOWPARAMS.py -P /fscnn && \
	
    wget https://raw.githubusercontent.com/ruonanl/fscnn/master/sliding_window_helpers.py -P /fscnn
	
WORKDIR /root
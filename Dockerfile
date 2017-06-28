# Ubuntu 14.04, CUDA 8.0

FROM microsoft/cntk:2.0-gpu-python3.5-cuda8.0-cudnn5.1

# Get frcnn git repo and set up

RUN apt-get install git-all && \

    git clone https://github.com/ruonanl/ObjectDetectionUsingCntk /fscnn && \
	
    pip install -r /fscnn/resources/python35_64bit_requirements/requirements.txt
	
# prepare fscnn-specifics
	
RUN mkdir /fscnn/data/drone/testImages && \

    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/sampleimage/1000093223.jpg -P /fscnn/data/drone/testImages && \
	
    mkdir /fscnn/proc1/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/1/frcn_nn.model -P /fscnn/proc1/drone/models && \
	
    mkdir /fscnn/proc2/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/2/frcn_nn.model -P /fscnn/proc2/drone/models

FROM microsoft/cntk:2.0-cpu-python3.5

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

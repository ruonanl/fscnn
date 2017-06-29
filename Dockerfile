# Ubuntu 14.04, CUDA 8.0
#FROM microsoft/cntk:2.0-gpu-python3.5-cuda8.0-cudnn5.1

FROM microsoft/cntk:2.0-cpu-python3.5

###### Get frcnn git repo and set up

RUN apt-get -qq update

RUN sudo apt-get -qq -y install git-all
	
RUN wget https://bootstrap.pypa.io/get-pip.py && \

    python get-pip.py

RUN sudo apt-get -qq -y install cmake

RUN git clone https://github.com/ruonanl/ObjectDetectionUsingCntk /fscnn
	
#RUN pip install -r ./requirements.txt ## tentatively replaced by the lines below

RUN pip install scikit-learn

RUN pip install Pillow

RUN pip install future

RUN pip install EasyDict

RUN pip install opencv-python

#RUN pip install dlib ## tentatively removed


###### prepare fscnn-specifics resources
	
RUN mkdir -p /fscnn/data/drone/testImages && \

    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/sampleimage/1000093223.jpg -P /fscnn/data/drone/testImages && \
	
    mkdir -p /fscnn/proc1/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/1/frcn_nn.model -P /fscnn/proc1/drone/models && \
	
    mkdir -p /fscnn/proc2/drone/models && \
	
    wget https://droneimageadsgpu.blob.core.windows.net/fscnncntk2trainedmodels/2/frcn_nn.model -P /fscnn/proc2/drone/models
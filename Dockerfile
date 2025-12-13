# Use NVIDIA's CUDA 12.3.1 runtime image with Ubuntu 22.04 as the base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt update && apt upgrade -y

RUN apt-get install git -y

# Clone the SHARK repository from GitHub and install wget
RUN git clone https://github.com/nod-ai/SHARK.git &&\
    apt-get install wget 

RUN apt update && \
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc &&\
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list &&\
    apt update &&\
    apt install vulkan-sdk -y

# Install software-properties-common and gpg-agent, then add deadsnakes PPA for Python versions 3.11 and above
RUN apt-get update && \
    apt-get install -y software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install python3.11 -y python3.11-venv

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN ["/bin/bash"]

RUN cd SHARK &&\
    ./setup_venv.sh

# # Start interactive shell
# RUN apt-get install -y cuda-drivers

# Ensure the SHARK virtual environment is activated whenever starting a new shell
RUN cd SHARK &&\
    . shark.venv/bin/activate &&\
    echo ". SHARK/shark.venv/bin/activate" >> ~/.bashrc

CMD ["/bin/bash"]
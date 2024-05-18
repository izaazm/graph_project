FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get install -y python3-pip wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

RUN python3 -m pip install --upgrade pip
                
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy


# set environment variable
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# setup conda virtual environment
COPY ./requirements.txt /tmp/requirements.txt
RUN conda update conda \
    && conda env create --name ingram --file /tmp/requirements.txt

RUN echo "conda activate ingram" >> ~/.bashrc
ENV PATH /opt/conda/envs/ingram/bin:$PATH
ENV CONDA_DEFAULT_ENV $ingram

# install pytorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3" ]
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install required packages
RUN apt-get update --fix-missing
RUN apt-get install -y python3-pip wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN python3 -m pip install --upgrade pip
                
# install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set environment variable
ENV PATH /opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# setup conda virtual environment
COPY ./environment.yml /tmp/environment.yml
RUN conda update conda \
    && conda env create --name ingram --file /tmp/environment.yml

RUN echo "conda activate ingram" >> ~/.bashrc
ENV PATH /opt/conda/envs/ingram/bin:$PATH
ENV CONDA_DEFAULT_ENV $ingram

# install pytorch
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Set the working directory
WORKDIR /app
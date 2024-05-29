FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

#RUN add-apt-repository ppa:savoury1/blender
RUN  apt-get update && apt-get install -y \
    software-properties-common python3.10 python3-pip
RUN ls
RUN apt-get update && apt-get install --allow-downgrades -y \
    ffmpeg \
    curl \
    gcc \
    git-lfs \
    wget \
    unzip \
    python-is-python3 \
    libglfw3-dev \
    libgles2-mesa-dev \
    libsm6 \
    libxext6 \
    libxrender-dev

# Download Blender 3.6
RUN wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz

# Extract Blender
RUN tar -xvf blender-3.6.0-linux-x64.tar.xz && rm blender-3.6.0-linux-x64.tar.xz

# Move Blender to /opt
RUN mv blender-3.6.0-linux-x64 /opt/blender-3.6

# Create a symlink to make Blender accessible from anywhere
RUN ln -s /opt/blender-3.6/blender /usr/local/bin/blender

#RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/lib/python3.8 1
RUN python --version
COPY . .
RUN ls
#RUN python --version
RUN pip install --upgrade pip
RUN pip install gdown firebase-admin
#RUN pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
RUN git lfs install
RUN git clone https://github.com/samiul272/champ.git
WORKDIR /champ
COPY . .
RUN pip install -r requirements.txt
RUN git clone https://huggingface.co/fudan-generative-ai/champ pretrained_models
RUN git clone https://github.com/shubham-goel/4D-Humans.git && cd 4D-Humans && pip install torch && pip install -e \
    .[all] && mkdir data && cd data && gdown --id 1L5WnG9MremVgEoU-RODndjCW9gss6xZu
RUN pip install git+https://github.com/facebookresearch/detectron2
RUN mkdir -p annotator/ckpts/ && cd annotator/ckpts && gdown --id  12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2 && \
    gdown --id 1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI
RUN python -m scripts.pretrained_models.download --hmr2 && python -m scripts.pretrained_models.download --detectron
RUN git clone https://github.com/IDEA-Research/DWPose.git DWPose
RUN cd ~/.cache && mkdir -p 4DHumans/data && cd 4DHumans/data && gdown --id 1L5WnG9MremVgEoU-RODndjCW9gss6xZu

COPY . .
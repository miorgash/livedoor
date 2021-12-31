FROM ubuntu:18.04
RUN : "essential" && \
    apt update -y && \
    apt upgrade -y && \
    apt install sudo apt-utils vim git -y && \
    : "japanese setting for os" && \
    apt-get install -y language-pack-ja-base language-pack-ja && \
    : "japanese setting for jupyter" && \
    sudo apt-get install -y fonts-ipaexfont && \
    : "python" && \
    apt install python3.7 python3.7-dev python3-pip python3.7-venv -y && \
    python3.7 -m pip install -U pip && \
    : "required by mecab-python3, Tensorflow etc." && \
    apt install swig -y && \
    python3.7 -m pip install wheel && \
    python3.7 -m pip install -U setuptools && \
    python3.7 -m pip install -U six
ENV LANG=ja_JP.UTF-8
RUN : "image peculiar settings" && \
    : "MeCab and dictionaries": && \
    apt install mecab libmecab-dev mecab-ipadic-utf8 -y && \
    apt install unidic-mecab -y && \
    : "update-alternatives --config mecab-dictionary # interactive check command" && \
    apt install git make curl xz-utils file unzip -y && \
    git clone --depth 1 https://github.com/neologd/mecab-unidic-neologd /tmp/mecab-unidic-neologd
WORKDIR /tmp/mecab-ipadic-neologd
RUN : "ipadic-neologd" && \
    git clone https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    sudo bin/install-mecab-ipadic-neologd -y -n
WORKDIR /tmp/mecab-unidic-neologd
RUN : "unidic-neologd" && \
    ./bin/install-mecab-unidic-neologd -n -y
COPY ./requirements.txt /var/requirements.txt
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=$PATH:/root/.cargo/bin
RUN : "Default kernel" && \
    pip install -U pip && \
    : "torch; torchtext が torch に依存するので requirements.txt を用いた install より先に実行する" && \
    python3.7 -m pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r /var/requirements.txt
RUN : "create dir" && \
    mkdir -p /data/

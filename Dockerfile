FROM ubuntu:22.04

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    vim \
    cmake \
    gfortran \
    libgsl27 libgsl-dev \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    gcc \
    gawk \
    bison \
    libblas3 \
    libblas-dev \
    liblapack3 \
    liblapack-dev \
    libatlas3-base \
    libatlas-base-dev \
    libc6 libc6-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/JohannesBuchner/MultiNest.git && \
    cd MultiNest/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/MultiNest && \
    make && \
    cd ../.. && rm -rf MultiNest

ENV LD_LIBRARY_PATH=/opt/MultiNest/lib:$LD_LIBRARY_PATH

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip3 install numpy pandas scipy toml tabulate plotly matplotlib corner astropy emcee pymultinest ipython progressbar -i https://pypi.tuna.tsinghua.edu.cn/simple/

WORKDIR /workspace

COPY ./bayspec /workspace/bayspec

CMD ["/bin/bash"]
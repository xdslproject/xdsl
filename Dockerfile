FROM ubuntu:latest

# Install xz-utils
RUN apt-get update && apt-get install -y \
    wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the official LLVM 19.1.7 binary
WORKDIR /tmp
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-19.1.7/LLVM-19.1.7-Linux-X64.tar.xz && \
    tar xf LLVM-19.1.7-Linux-X64.tar.xz -C /usr/local --strip-components=1 && \
    rm LLVM-19.1.7-Linux-X64.tar.xz

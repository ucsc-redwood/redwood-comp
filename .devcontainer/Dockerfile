# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install xmake, 7zip, and just
RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    curl \
    wget \
    vim \
    git \
    # xmake related
    xmake \
    p7zip-full \
    # my projects related
    libbenchmark-dev \
    cmake \
    # development related
    clangd \
    clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install Vulkan SDK
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list && \
    apt-get update && apt-get install -y vulkan-sdk && \
    rm -rf /var/lib/apt/lists/*

RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the ubuntu user to install just in ~/bin
USER ubuntu
RUN mkdir -p ~/bin && \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin && \
    echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc

# Default shell for ubuntu
SHELL ["/bin/bash", "-l"]

# Set working directory for the user
WORKDIR /home/ubuntu

# Default command
CMD ["/bin/bash"]

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

ARG TF_BRANCH=r1.10
ARG BAZEL_VERSION=0.15.0
ARG TF_AVAILABLE_CPUS=7

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        golang \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-pip \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
        pip setuptools

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        keras_applications==1.0.5 \
        keras_preprocessing==1.0.3 \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        wheel \
        && \
    python -m ipykernel.kernelspec


# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    wget --quiet https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    wget --quiet https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Download and build TensorFlow.
WORKDIR /
RUN git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout ${TF_BRANCH}
WORKDIR /tensorflow

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON=python \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
    PYTHON_BIN_PATH=/usr/bin/python \
    PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages \
    TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=9.2 \
    TF_CUDA_COMPUTE_CAPABILITIES=5.0,5.2,6.0,6.1,7.0 \
    TF_CUDNN_VERSION=7
RUN ./configure

# Build and Install TensorFlow.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    bazel build -c opt \
                --config=cuda \
                --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
                --jobs=${TF_AVAILABLE_CPUS} \
                tensorflow/tools/pip_package:build_pip_package && \
    mkdir /pip_pkg && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /pip_pkg && \
    pip --no-cache-dir install --upgrade /pip_pkg/tensorflow-*.whl && \
    rm -rf /pip_pkg && \
    rm -rf /root/.cache
# Clean up pip wheel and Bazel cache when done.

# Install requirements for MSc project
ADD ./requirements.txt ./requirements.txt
RUN /opt/conda/bin/pip install --trusted-host pypi.python.org -r requirements.txt
RUN git clone https://github.com/yngvem/scinets.git


# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

WORKDIR /root
CMD ["/bin/bash"]

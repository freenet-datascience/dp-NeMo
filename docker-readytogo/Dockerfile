#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's PyTorch container image.
# We further add everything needed for NeMo & parlance transcription
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#

FROM ubuntu:jammy-20221003
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3-torch python3-pip python3-dev git wget build-essential --no-install-recommends && \
     apt-get -yq install swig build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev ffmpeg libpcre3 libpcre3-dev --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip 
RUN pip3 install wheel cython pesq --no-cache-dir


WORKDIR /workspace/

# install packages required for training language models

WORKDIR /workspace/

RUN git clone https://github.com/NVIDIA/NeMo.git
RUN pip3 install -r /workspace/NeMo/requirements/requirements.txt --no-cache-dir
RUN pip3 install -r /workspace/NeMo/requirements/requirements_common.txt --no-cache-dir
RUN pip3 install -r /workspace/NeMo/requirements/requirements_asr.txt --no-cache-dir
RUN pip3 install -r /workspace/NeMo/requirements/requirements_lightning.txt --no-cache-dir
# 
# install nemo
WORKDIR /workspace/NeMo/
RUN pip3 install -e .



WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm
RUN git clone --single-branch -b ctc-decoders https://github.com/NVIDIA/OpenSeq2Seq
RUN mv OpenSeq2Seq/decoders .
RUN rm -r OpenSeq2Seq
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/decoders
# kenlm
RUN git clone https://github.com/kpu/kenlm.git
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/decoders/kenlm
RUN mkdir -p build
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/decoders/kenlm/build
RUN cmake ..
RUN make -j 4
# openfst
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm/decoders/
RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
RUN tar -xzvf openfst-1.6.3.tar.gz
# ThreadPool
RUN git clone https://github.com/progschj/ThreadPool.git

RUN python3 setup.py install --num_processes 4
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm

# these packages are required for diarization
RUN python3 -m pip install dill levenshtein  --no-cache-dir

# install parlance
WORKDIR /workspace/NeMo/

RUN git clone --recursive https://github.com/parlance/ctcdecode.git
WORKDIR /workspace/NeMo/ctcdecode/
RUN pip3 install --upgrade pip && \
  pip3 install .
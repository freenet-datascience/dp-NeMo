#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's PyTorch container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
# Magnus: This works, but does not yet include the pytorch audio installation
FROM nvcr.io/nvidia/pytorch:22.08-py3

#  #
COPY reinstall.sh /tmp/
COPY setup.cfg /tmp/
COPY setup.py  /tmp/
COPY nemo /tmp/nemo/
COPY requirements /tmp/requirements/
WORKDIR /tmp/
RUN ./reinstall.sh
WORKDIR /workspace/

# install packages required for training language models
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -yq install swig build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev ffmpeg libpcre3 libpcre3-dev 
WORKDIR /workspace/

RUN git clone https://github.com/NVIDIA/NeMo.git
# install decoders
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

RUN python3.8 setup.py install --num_processes 4
WORKDIR /workspace/NeMo/scripts/asr_language_modeling/ngram_lm

# these packages are required for diarization
RUN python3.8 -m pip install pyctcdecode pypi-kenlm arpa levenshtein
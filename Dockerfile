FROM nvidia/cuda:11.0-devel-ubuntu20.04

# CMD nvidia-smi

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update
RUN apt-get -y install curl wget git python python3 python3-pip gfortran
RUN pip3 install numpy

WORKDIR /staging
RUN git clone https://github.com/Jokeren/spack.git && cd spack && git pull origin && git checkout 58a919d1677e0b172ec250b8dd84a1eb32efa366
ENV SPACK_ROOT=/staging/spack
RUN ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/docker-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/interactive-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/spack-env
SHELL ["docker-shell"]
ENTRYPOINT ["/bin/bash", "/staging/spack/share/spack/docker/entrypoint.bash"]

RUN spack spec dyninst@cgo
RUN spack install --only dependencies hpctoolkit ^dyninst@cgo ^elfutils
RUN spack install gcc@7.3.0
RUN spack load gcc@7.3.0
RUN which gcc

RUN git clone --recursive https://github.com/Jokeren/GPA.git
WORKDIR GPA
RUN mkdir build
RUN ./bin/install.sh $(pwd)/build $(spack find --path boost | tail -n 1 | cut -d ' ' -f 3 | sed 's,/*[^/]\+/*$,,')

ENTRYPOINT []

ENV CUDA_VISIBLE_DEVICES=0
ENV PATH=$(pwd)/build/bin:${PATH}
ENV PATH=$(pwd)/build/hpctoolkit/bin:${PATH}
CMD git pull origin master && ./bin/bench.sh -m bench && ./bin/bench.sh -m advise

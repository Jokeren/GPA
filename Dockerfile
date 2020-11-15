FROM nvidia/cuda:11.0-devel-ubuntu20.04

# CMD nvidia-smi

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update
RUN apt-get -y install curl git python gfortran wget

WORKDIR /staging
RUN git clone https://github.com/spack/spack.git && cd spack && git checkout fafff0c6c0142e62e0f6b65b1d53ea58feb7fc7a
ENV SPACK_ROOT=/staging/spack
RUN ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/docker-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/interactive-shell \
 && ln -s $SPACK_ROOT/share/spack/docker/entrypoint.bash \
          /usr/local/bin/spack-env
SHELL ["docker-shell"]
ENTRYPOINT ["/bin/bash", "/staging/spack/share/spack/docker/entrypoint.bash"]

RUN spack spec hpctoolkit
RUN spack install --only dependencies hpctoolkit ^dyninst@master

RUN git clone --recursive https://github.com/Jokeren/GPA.git
WORKDIR GPA
RUN mkdir build
RUN ./bin/install.sh $(pwd)/build $(spack find --path boost | tail -n 1 | cut -d ' ' -f 3 | sed 's,/*[^/]\+/*$,,')
ENV PATH=$(pwd)/build/bin:${PATH}
ENV CUDA_VISIBLE_DEVICES=0
CMD ./bin/bench.sh

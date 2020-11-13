FROM nvidia/cuda:11.1-devel-ubuntu20.04

# CMD nvidia-smi

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update
RUN apt-get -y install curl git python gfortran

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

WORKDIR /staging
COPY hpctoolkit ./hpctoolkit
WORKDIR hpctoolkit
WORKDIR build

RUN ../configure --prefix=/opt/hpctoolkit \
                 --with-cuda=/usr/local/cuda \
                 --with-cupti=/usr/local/cuda \
                 --with-spack=$(spack find --path boost | tail -n 1 | cut -d ' ' -f 3 | sed 's,/*[^/]\+/*$,,')
RUN make
RUN make install
ENV HPCTOOLKIT_ROOT=/opt/hpctoolkit
ENV PATH=$HPCTOOLKIT_ROOT/bin:$PATH

WORKDIR /workspace
COPY GPA-Benchmark ./GPA-Benchmark
WORKDIR GPA-Benchmark
WORKDIR gpa-minimod-artifacts
CMD TARGET=cuda_smem_u_s_opt-gpu COMPILER=nvcc make clean all \
 && ./main_cuda_smem_u_s_opt-gpu_nvcc \
 && hpcrun -e gpu=nvidia,pc -t ./main_cuda_smem_u_s_opt-gpu_nvcc \
 && hpcstruct ./main_cuda_smem_u_s_opt-gpu_nvcc \
 && hpcstruct --gpucfg yes hpctoolkit-main_cuda_smem_u_s_opt-gpu_nvcc-measurements \
 && hpcprof -S main_cuda_smem_u_s_opt-gpu_nvcc.hpcstruct -I ./+ hpctoolkit-main_cuda_smem_u_s_opt-gpu_nvcc-measurements


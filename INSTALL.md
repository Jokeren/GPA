## Install spack

```bash
git clone https://github.com/spack/spack.git
export SPACK_ROOT=/path/to/spack
source ${SPACK_ROOT}/share/spack/setup-env.sh
```

## Install dependencies

```bash
spack spec hpctoolkit
spack install --only dependencies hpctoolkit ^dyninst@master
```

## Install hpctoolkit

### hpctoolkit

```bash
cd /path/to/hpctoolkit
mkdir build && cd build
# Tip: check spack libraries' root->spack find --path.  
# For example: --with-spack=/home/username/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/
../configure --prefix=/path/to/hpctoolkit/installation --with-cuda=/usr/local/cuda-11.0 --with-cupti=/path/to/cupti/root --with-spack=/path/to/spack/libraries/root
make install -j8
```

### hpcviewer

http://hpctoolkit.org/download/hpcviewer/

### Add to environment

Add following lines into your `.bashrc` file and source it.

```bash
export HPCTOOLKIT=/path/to/hpctoolkit
export PATH=$HPCTOOLKIT/bin/:$PATH
```

## Test

```bash
cd ./GPA-Benchmark/ExaTENSOR
make
hpcrun -e gpu=nvidia,pc ./main
hpcstruct --gpucfg yes hpctoolkit-main-measurement
hpcstruct ./main
hpcprof -S main.hpcstruct hpctoolkit-main-measurement
hpcviewer hpctoolkit-main-database
```

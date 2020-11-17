#!/bin/bash

if [ ! -d GPA-Benchmark/data ]; then
  cd ./GPA-Benchmark
  ./get_data.sh
  cd ..
fi

python3 ./python/bench.py ${@:1}

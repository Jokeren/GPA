#!/bin/bash

if [ ! -d GPA-Benchmark/data ]; then
  cd ./GPA-Benchmark
  ./get_data.sh
  cd ..
fi

if [ $# -eq 1 ]; then
  python3 ./python/bench.py $1
else
  python3 ./python/bench.py
fi

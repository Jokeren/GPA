#!/bin/bash

DIR=""

if [ $# -le 1 ]; then
  echo "Wrong options"
  echo "./bench.sh <gpa prefix> [benchmark]"
else
  $DIR=$1
fi

cd $DIR
if [ ! -d GPA-Benchmark/data ]; then
  cd $DIR/GPA-Benchmark
  ./get_data.sh
  cd ..
fi

if [ $# -eq 2]; then
  python $DIR/python/bench.py $2
else
  python $DIR/python/bench.py
fi

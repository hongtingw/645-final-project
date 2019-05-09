#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
echo "Preparing MNIST test data..."
mkdir data
cd data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
cd ..
echo "Preparing LeNet-5 model..."
pip install tensorflow==2.0.0-alpha0
python train_lenet.py --output_dir model
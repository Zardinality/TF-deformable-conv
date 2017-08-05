# TF-deformable-conv

This is a repository for a [Deformable Convolution](https://arxiv.org/abs/1703.06211) operation in Tensorflow. This repo largely borrows cuda codes from [original implementation](https://github.com/msracver/Deformable-ConvNets).

### Check [here](https://github.com/Zardinality/TF_Deformable_Net) for a inplementation of Deformable net in tensorflow.

## Prerequisite

Tensorflow(with GPU configured)

Cuda 8.0

g++ 4.9.2

*Note*: Only tested on platform where corresponding version of g++ and cuda installed, g++5 might encounter `undefined symbol` problem, It's suggested to reinstall g++4.9 to solve this problem, as pointed out by @cotrane in [this issue](https://github.com/Zardinality/TF-deformable-conv/issues/1). Here are the steps:

- installing gcc-4.9 and g++-4.9

- changing `nvcc_compile.sh` to:

  ```shell
  nvcc -std=c++11 -ccbin=/usr/bin/g++-4.9 -c -o deform_conv.cu.o deform_conv.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-8.0/lib64/ --expt-relaxed-constexpr
  ```

- and changing `g++_complie.sh` to:

  ```shell
  g++-4.9 -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I TF_INC -fPIC -lcudart -L CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0
  ```

## Usage

1. Set up `TF_INC` and `CUDA_HOME`, where `TF_INC` can be set up as  `TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')`. Make sure `CUDA_HOME` be the path where cuda is installed, such as default: `/usr/local/cuda`.
2. Build the op. If you have Tensorflow source installed, you could copy all cpp files contained in `./lib` and `BUILD` to `$(Tensorflow_source_dir)/tensoflow/core/user_ops`, then run `bazel build --config=opt --config=cuda //tensorflow/core/user_ops:deform_conv.so` in  `$(Tensorflow_source_dir)`. If not, run `./lib/nvcc_complie.sh`and `./lib/g++_complie.sh` in sequence to build `deform_conv.so`. (If `cuda_config.h` is reported to be missed, check [here](https://github.com/Zardinality/TF-deformable-conv/issues/1))
3. `import lib.deform_conv_op as deform_conv_op` in your python script (make sure PYTHON_PATH was set currectly).


## Demo

A simple WGAN script trained on MNIST, to validated the backpropagation.

![](https://ws4.sinaimg.cn/large/006tKfTcgy1fgspsomt2xj30da0d1abl.jpg)

Since offset mostly stays between -1 and 1 there is no need to visualize it. Considering the simplicity of discriminator task, I'm not suprised about it. Might considering bring scaled MNIST in and pretrain regular conv part or change the initializer of offset conv to random normal to make deform matters.

## TODO

- [x] Basic test with original implementation.
- [x] --Make sure gradient work.(weird bug happened, data grad used to be correct except for first time calculated, now in my test it works normal, but if you find any bug just open an issue)
- [x] Simple benchmark.


- [x] Some demo and visualization.
- [ ] Backward time costs too much.
- [ ] Other ops.

## Benchmark

Benchmark script was borrowed from [here](https://github.com/soumith/convnet-benchmarks/blob/master/tensorflow/benchmark_alexnet.py). The forward time is fine, for 100x3x224x224 data, it runs about in 0.077s. But backward time generaly undesired, it cost 0.558s to run a batch of same data. Note I write all backward of three inputs(data, offset, kernels) together, rather than like many tensorflow conv ops spliting input_backwards and kernel_backwards to two ops, so this might be one of the reason. In addition, because  sometimes I find it hard to manipulate `tensorflow::Tensor` , I write a simple cuda kernel that does nothing but add one tensor to another, for accumulating gradients along batch in kernel gradient implementation, don't know whether it affects performance.
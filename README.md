# TF-deformable-conv

This is a repository for a [Deformable Convolution](https://arxiv.org/abs/1703.06211) operation in Tensorflow. This repo largely borrows cuda codes from [original implementation](https://github.com/msracver/Deformable-ConvNets).

## Prerequisite

Tensorflow(with GPU configured)

Cuda 8.0

g++ 4.9.2

*Note*: Only tested on platform where corresponding version of g++ and cuda installed, other version might generally be fine, but may need to modify the compile script.

## Usage

1. Set up `TF_INC` and `CUDA_HOME`, where `TF_INC` can be set up as  `TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')`. Make sure `CUDA_HOME` be the path where cuda is installed, such as default: `/usr/local/cuda`.
2. Build the op. If you have Tensorflow source installed, you could copy all cpp files contained in `./lib` and `BUILD` to `$(Tensorflow_source_dir)/tensoflow/core/user_ops`, then run `bazel build --config=opt --config=cuda //tensorflow/core/user_ops:deform_conv.so` in  `$(Tensorflow_source_dir)`. If not, run `./lib/nvcc_complie.sh`and `./lib/g++_complie.sh` in sequence to build `deform_conv.so`.
3. `import lib.deform_conv_op as deform_conv_op` in your python script (make sure PYTHON_PATH was set currectly).


## TODO

- [x] Basic test with original implementation.
- [x] --Make sure gradient work.(weird bug happened, data grad used to be correct except for first time calculated, now in my test it works normal, but if you find any bug just open an issue)
- [x] Simple benchmark.


- [ ] Some demo and visualization.
- [ ] Backward time costs too much.
- [ ] Other ops.

## Benchmark

Benchmark script was borrowed from [here](https://github.com/soumith/convnet-benchmarks/blob/master/tensorflow/benchmark_alexnet.py). The forward time is fine, for 100x3x224x224 data, it runs about in 0.077s. But backward time generaly undesired, it cost 0.558s to run a batch of same data. Note I write all backward of three inputs(data, offset, kernels) together, rather than like many tensorflow conv ops spliting input_backwards and kernel_backwards to two ops, so this might be one of the reason. In addition, because  sometimes I find it hard to manipulate `tensorflow::Tensor` , I write a simple cuda kernel that does nothing but add one tensor to another, for accumulating gradients along batch in kernel gradient implementation, don't know whether it affects performance.
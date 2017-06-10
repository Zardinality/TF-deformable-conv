This is a repository for a [Deformable Convolution](https://arxiv.org/abs/1703.06211) operation in Tensorflow. This repo largely borrows cuda codes from [original implementation](https://github.com/msracver/Deformable-ConvNets).

## Prerequisite

Tensorflow(with GPU configured)

cuda 8.0

g++ 4.9.2



*Note*: Only tested on platform where corresponding version of g++ and cuda installed, other version might generally be fine, but may need to modify the compile script.

## Usage

1. Set up `TF_INC` and `CUDA_HOME`, where `TF_INC` can be set up as  `TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')`. Make sure `CUDA_HOME` be the path where cuda is installed, such as default: `/usr/local/cuda`.
2. Build the op. If you have Tensorflow source installed, you could copy all cpp files contained in `./lib` and `BUILD` to `$(Tensorflow_source_dir)/tensoflow/core/user_ops`, then run `bazel build --config=opt --config=cuda //tensorflow/core/user_ops:deform_conv.so` in  `$(Tensorflow_source_dir)`. If not, run `./lib/nvcc_complie.sh`and `./lib/g++_complie.sh` in sequence to build `deform_conv.so`.
3. `import lib.deform_conv_op as deform_conv_op` in your python script (make sure PYTHON_PATH was set currectly).




## TODO

- [x] Basic test with original implementation.
- [x] Make sure gradient work.
- [x] Simple benchmark


- [ ] Some demo and visualization
- [ ] Backward time costs too much
- [ ] Other ops


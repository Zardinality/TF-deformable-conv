TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC
if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi
g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0

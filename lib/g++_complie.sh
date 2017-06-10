g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0

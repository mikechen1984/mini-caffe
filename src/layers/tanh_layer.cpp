// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <vector>

#include "./tanh_layer.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_tanh_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

#ifdef USE_CUDA
STUB_CPU(TanHLayer);
#else

void TanHLayer::Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}

STUB_GPU(TanHLayer);
#endif
// Creator

}  // namespace caffe

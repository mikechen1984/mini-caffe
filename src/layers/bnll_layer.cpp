#include <algorithm>
#include <vector>

#include "./bnll_layer.hpp"

namespace caffe {
#ifdef USE_CUDA
STUB_CPU(BNLLLayer);
#else
void BNLLLayer::Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] > 0 ?
        bottom_data[i] + log(1. + exp(-bottom_data[i])) :
        log(1. + exp(bottom_data[i]));
  }
}

STUB_GPU(BNLLLayer);
#endif



}  // namespace caffe

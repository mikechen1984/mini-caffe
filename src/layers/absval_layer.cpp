#include <vector>

#include "./absval_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {
#ifdef USE_CUDA
STUB_CPU(AbsValLayer);
#else
void AbsValLayer::Forward_cpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  const int count = top[0]->count();
  real_t* top_data = top[0]->mutable_cpu_data();
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

STUB_GPU(AbsValLayer);
#endif



}  // namespace caffe

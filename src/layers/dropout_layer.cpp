// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "./dropout_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {
#ifdef USE_CUDA
STUB_CPU(DropoutLayer);
#else
void DropoutLayer::Forward_cpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

STUB_GPU(DropoutLayer);
#endif



}  // namespace caffe

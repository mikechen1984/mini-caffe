#include <cmath>
#include <vector>

#include "./sigmoid_layer.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_sigmoid_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {

inline real_t sigmoid(real_t x) {
  return 1. / (1. + exp(-x));
}
#ifdef USE_CUDA
STUB_CPU(SigmoidLayer);
#else

void SigmoidLayer::Forward_cpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

STUB_GPU(SigmoidLayer);
#endif

// Creator

}  // namespace caffe

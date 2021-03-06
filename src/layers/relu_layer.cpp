#include <algorithm>
#include <vector>

#include "./relu_layer.hpp"

#ifdef USE_CUDNN
#include "./cudnn/cudnn_relu_layer.hpp"
#endif  // USE_CUDNN

namespace caffe {
#ifdef USE_CUDA
STUB_CPU(ReLULayer);
#else
void ReLULayer::Forward_cpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->cpu_data();
  real_t* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  real_t negative_slope = this->layer_param_.relu_param().negative_slope();
  if (std::abs(negative_slope) < 1e-6) {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], static_cast<real_t>(0));
    }
  }
  else {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], static_cast<real_t>(0))
        + negative_slope * std::min(bottom_data[i], static_cast<real_t>(0));
    }
  }
}

STUB_GPU(ReLULayer);
#endif

// Creator


}  // namespace caffe

/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#include "./layer_factory.hpp"
#include "layers/elu_layer.hpp"
#include "layers/prior_box_layer.hpp"
#include "layers/proposal_layer.hpp"
#include "layers/embed_layer.hpp"
#include "layers/absval_layer.hpp"
#include "layers/argmax_layer.hpp"
#include "layers/batch_norm_layer.hpp"
#include "layers/filter_layer.hpp"
#include "layers/psroi_pooling_layer.hpp"
#include "layers/exp_layer.hpp"
#include "layers/reduction_layer.hpp"

#include "layers/tile_layer.hpp"
#include "layers/flatten_layer.hpp"
#include "layers/bias_layer.hpp"
#include "layers/relu_layer.hpp"
#include "layers/reshape_layer.hpp"
#include "layers/input_layer.hpp"
#include "layers/bn_layer.hpp"

#include "layers/inner_product_layer.hpp"
#include "layers/roi_pooling_layer.hpp"
#include "layers/bnll_layer.hpp"
#include "layers/log_layer.hpp"
#include "layers/scale_layer.hpp"
#include "layers/lrn_layer.hpp"
#include "layers/concat_layer.hpp"
#include "layers/shuffle_channel_layer.hpp"

#include "layers/conv_dw_layer.hpp"
#include "layers/mvn_layer.hpp"
#include "layers/sigmoid_layer.hpp"
#include "layers/conv_layer.hpp"
#include "layers/neuron_layer.hpp"
#include "layers/slice_layer.hpp"
#include "layers/crop_layer.hpp"
#include "layers/normalize_layer.hpp"
#include "layers/parameter_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/deconv_layer.hpp"
#include "layers/split_layer.hpp"
#include "layers/spp_layer.hpp"
#include "layers/permute_layer.hpp"
#include "layers/detection_output_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/tanh_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "layers/power_layer.hpp"
#include "layers/threshold_layer.hpp"
#include "layers/eltwise_layer.hpp"
#include "layers/prelu_layer.hpp"
namespace caffe {

template<>
shared_ptr<Layer> CreateLayer<BNLayer>(const LayerParameter &param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNBNLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new BNLayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<TanHLayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNTanHLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new TanHLayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<ConvolutionLayer>(const LayerParameter &param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    ConvolutionParameter conv_param = param.convolution_param();
    if (conv_param.group() == conv_param.num_output()) {  // depthwise
      return shared_ptr<Layer>(new ConvolutionDepthwiseLayer(param));
    }
    bool use_dilation = false;
    for (int i = 0; i < conv_param.dilation_size(); ++i) {
      if (conv_param.dilation(i) > 1) {
        use_dilation = true;
      }
    }
    if (!use_dilation) {
      return shared_ptr<Layer>(new CuDNNConvolutionLayer(param));
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new ConvolutionLayer(param));
}


template<>
shared_ptr<Layer> CreateLayer<DeconvolutionLayer>(const LayerParameter &param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    ConvolutionParameter conv_param = param.convolution_param();
    bool use_dilation = false;
    for (int i = 0; i < conv_param.dilation_size(); ++i) {
      if (conv_param.dilation(i) > 1) {
        use_dilation = true;
      }
    }
    if (!use_dilation) {
      return shared_ptr<Layer>(new CuDNNDeconvolutionLayer(param));
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new DeconvolutionLayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<LRNLayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    LRNParameter lrn_param = param.lrn_param();
    if (lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer>(new CuDNNLCNLayer(param));
    }
    else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer>(new LRNLayer(param));
      }
      else {
        return shared_ptr<Layer>(new CuDNNLRNLayer(param));
      }
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new LRNLayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<PoolingLayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    if (param.top_size() == 1) {
      return shared_ptr<Layer>(new CuDNNPoolingLayer(param));
    }
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new PoolingLayer(param));
}


template<>
shared_ptr<Layer> CreateLayer<ReLULayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
     return shared_ptr<Layer>(new CuDNNReLULayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new ReLULayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<SigmoidLayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNSigmoidLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new SigmoidLayer(param));
}

template<>
shared_ptr<Layer> CreateLayer<SoftmaxLayer>(const LayerParameter& param) {
#ifdef USE_CUDNN
  if (Caffe::mode() == Caffe::GPU) {
    return shared_ptr<Layer>(new CuDNNSoftmaxLayer(param));
  }
#endif  // USE_CUDNN
  return shared_ptr<Layer>(new SoftmaxLayer(param));
}

        

#define REGISTER_LAYER_CREATOR(type, creator)     \
{                                                 \
    caffe::LayerRegister layer_register(#type, creator); \
}                                                 \

#define REGISTER_LAYER_CLASS(type) REGISTER_LAYER_CREATOR(type, CreateLayer<type##Layer>)

void initLayerRegistry(){
  REGISTER_LAYER_CLASS(ReLU)
  REGISTER_LAYER_CLASS(BN)
  REGISTER_LAYER_CLASS(TanH)
  REGISTER_LAYER_CLASS(Convolution)
  REGISTER_LAYER_CLASS(Deconvolution)
  REGISTER_LAYER_CLASS(LRN)
  REGISTER_LAYER_CLASS(Pooling)
  REGISTER_LAYER_CLASS(Sigmoid)
  REGISTER_LAYER_CLASS(Softmax)
  REGISTER_LAYER_CLASS(ELU)
  REGISTER_LAYER_CLASS(Embed)
  REGISTER_LAYER_CLASS(Exp)
  REGISTER_LAYER_CLASS(Filter)
  REGISTER_LAYER_CLASS(Flatten)
  REGISTER_LAYER_CLASS(InnerProduct)
  REGISTER_LAYER_CLASS(Input)
  REGISTER_LAYER_CLASS(Log)
  REGISTER_LAYER_CLASS(MVN)
  REGISTER_LAYER_CLASS(Normalize)
  REGISTER_LAYER_CLASS(Parameter)
  REGISTER_LAYER_CLASS(Permute)
  REGISTER_LAYER_CLASS(Power)
  REGISTER_LAYER_CLASS(PReLU)
  REGISTER_LAYER_CLASS(PriorBox)
  REGISTER_LAYER_CLASS(Proposal)
  REGISTER_LAYER_CLASS(PSROIPooling)
  REGISTER_LAYER_CLASS(Reduction)
  REGISTER_LAYER_CLASS(Reshape)
  REGISTER_LAYER_CLASS(ROIPooling)
  REGISTER_LAYER_CLASS(Scale)
  REGISTER_LAYER_CLASS(ShuffleChannel)
  REGISTER_LAYER_CLASS(Slice)
  REGISTER_LAYER_CLASS(Split)
  REGISTER_LAYER_CLASS(SPP)
  REGISTER_LAYER_CLASS(Threshold)
  REGISTER_LAYER_CLASS(Tile)
  REGISTER_LAYER_CLASS(ArgMax)
  REGISTER_LAYER_CLASS(BatchNorm)
  REGISTER_LAYER_CLASS(Bias)
  REGISTER_LAYER_CLASS(BNLL)
  REGISTER_LAYER_CLASS(Concat)
  REGISTER_LAYER_CLASS(ConvolutionDepthwise)
  REGISTER_LAYER_CLASS(Crop)
  REGISTER_LAYER_CLASS(DetectionOutput)
  REGISTER_LAYER_CLASS(Eltwise)
  REGISTER_LAYER_CLASS(Dropout)
  REGISTER_LAYER_CLASS(AbsVal)  
}

}
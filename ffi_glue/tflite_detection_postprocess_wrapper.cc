#include "edge-impulse-sdk/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
TfLiteRegistration* Register_TFLite_Detection_PostProcess() {
    return tflite::Register_DETECTION_POSTPROCESS();
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite

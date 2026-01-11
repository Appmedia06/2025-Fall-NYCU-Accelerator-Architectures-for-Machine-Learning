/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>
#include "cfu.h"
#include <cstdio>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const float val = input_data[i];
    // Note that alpha might be > 1 or < 0, so we don't use std::max here.
    printf("LeakyRelu alpha = %f\n", (double)params.alpha);
    output_data[i] = val > 0 ? val : val * params.alpha;
    
    // output_data[i] = cfu_op0(5, val, 0);
  }
}

// inline int32_t my_MultiplyByQuantizedMultiplier(int32_t x,
//                                              int32_t quantized_multiplier,
//                                              int shift) {                                        
//   // using gemmlowp::RoundingDivideByPOT;
//   // using gemmlowp::SaturatingRoundingDoublingHighMul;
//   int left_shift = shift > 0 ? shift : 0;
//   int right_shift = shift > 0 ? 0 : -shift;
//   int m = x * (1 << left_shift);
//   int32_t res1 = (int32_t)cfu_op0(5, m, quantized_multiplier);
//   return cfu_op0(6, res1, right_shift);

//   // return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
//   //                                x * (1 << left_shift), quantized_multiplier),
//   //                            right_shift);
// }

inline int32_t my_MultiplyByQuantizedMultiplier(int32_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {                                           
//   TFLITE_DCHECK(quantized_multiplier >= 0);
//   TFLITE_DCHECK(shift >= -31 && shift <= 30);

//   const int64_t total_shift = 31 - shift;
//   const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
//   int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
//   result = result >> total_shift;

//   TFLITE_DCHECK(result >= std::numeric_limits<int32_t>::min() &&
//                 result <= std::numeric_limits<int32_t>::max());
//   return static_cast<int32_t>(result);

  cfu_op0(8, x, 0);
  return cfu_op0(9, quantized_multiplier, shift);

}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = -128; // std::numeric_limits<T>::min();
  static const int32_t quantized_max = 127; // std::numeric_limits<T>::max();
  // printf("input_shape = %d, output_shape = %d, flat_size = %d\n", input_shape, output_shape, flat_size);
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value = input_data[i] - params.input_offset;
    int32_t unclamped_output;
    cfu_op0(8, input_value, params.output_offset);
    if (input_value >= 0) {
      unclamped_output = cfu_op0(9, params.output_multiplier_identity, params.output_shift_identity);
                        //  my_MultiplyByQuantizedMultiplier(
                        //      input_value, params.output_multiplier_identity,
                        //      params.output_shift_identity);
    } else {
      unclamped_output = cfu_op0(9, params.output_multiplier_alpha, params.output_shift_alpha);
                        //  my_MultiplyByQuantizedMultiplier(
                        //      input_value, params.output_multiplier_alpha,
                        //      params.output_shift_alpha);
    }
    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

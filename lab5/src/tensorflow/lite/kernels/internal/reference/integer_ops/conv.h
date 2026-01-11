/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "cfu.h"
#include "perf.h"
#include "playground_util/print_params.h"
#include <cstdio>

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
//   perf_enable_counter(6);
//   // Get parameters.
//   const int32_t input_offset = params.input_offset;  // r = s(q - Z)
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;

//   // Set min and max value of the output.
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency check.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
//   const int batches = MatchingDim(input_shape, 0, output_shape, 0);
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   // Check dimensions of the tensors.
//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int filter_input_depth = filter_shape.Dims(3);
//   const int groups = input_depth / filter_input_depth;
//   TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
//   const int filters_per_group = output_depth / groups;
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       const int in_y_origin = (out_y * stride_height) - pad_height;
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         const int in_x_origin = (out_x * stride_width) - pad_width;
//         for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
//           auto group = out_channel / filters_per_group;
//           int32_t acc = 0;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             const int in_y = in_y_origin + dilation_height_factor * filter_y;
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               const int in_x = in_x_origin + dilation_width_factor * filter_x;

//               // Zero padding by omitting the areas outside the image.
//               const bool is_point_inside_image =
//                   (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
//                   (in_y < input_height);

//               if (!is_point_inside_image) {
//                 continue;
//               }

//               for (int in_channel = 0; in_channel < filter_input_depth;
//                    ++in_channel) {
//                 int32_t input_val =
//                     input_data[Offset(input_shape, batch, in_y, in_x,
//                                       in_channel + group * filter_input_depth)];
//                 int32_t filter_val = filter_data[Offset(
//                     filter_shape, out_channel, filter_y, filter_x, in_channel)];
//                 // Accumulate with 32 bits accumulator.
//                 // In the nudging process during model quantization, we force
//                 // real value of 0.0 be represented by a quantized value. This
//                 // guarantees that the input_offset is a int8_t, even though
//                 // it is represented using int32_t. int32_t += int8_t *
//                 // (int8_t - int8_t) so the highest value we can get from each
//                 // accumulation is [-127, 127] * ([-128, 127] -
//                 // [-128, 127]), which is [-32512, 32512]. log2(32512)
//                 // = 14.98, which means we can accumulate at least 2^16
//                 // multiplications without overflow. The accumulator is
//                 // applied to a filter so the accumulation logic will hold as
//                 // long as the filter size (filter_y * filter_x * in_channel)
//                 // does not exceed 2^16, which is the case in all the models
//                 // we have seen so far.
//                 // TODO(b/174275578): Add a check to make sure the
//                 // accumulator depth is smaller than 2^16.
//                 acc += filter_val * (input_val + input_offset);
//               }
//             }
//           }

//           if (bias_data) {
//             acc += bias_data[out_channel];
//           }
//           acc = MultiplyByQuantizedMultiplier(
//               acc, output_multiplier[out_channel], output_shift[out_channel]);
//           acc += output_offset;
//           acc = std::max(acc, output_activation_min);
//           acc = std::min(acc, output_activation_max);
//           output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
//               static_cast<int8_t>(acc);
//         }
//       }
//     }
//   }
//   perf_disable_counter(6);
// }

// Fixed-point per-channel-quantization convolution reference kernel.
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
  
//   // Start Performance Counter 6
//   perf_enable_counter(6);

//   // Get parameters.
//   const int32_t input_offset = params.input_offset;
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;

//   // Set min and max value of the output.
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency check.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   // const int filter_input_depth = filter_shape.Dims(3); // Unused variable
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);

//   // -----------------------------------------------------------------------
//   // STATIC BUFFERS
//   // -----------------------------------------------------------------------
//   #define MAX_KERNEL_BUFFER_SIZE (2048 * 512)
//   #define MAX_IM2COL_BUFFER_SIZE (2048 * 512)
//   #define MAX_RESULT_BUFFER_SIZE (2048 * 512)

//   static int8_t filter_buffer[MAX_KERNEL_BUFFER_SIZE]; // Buffer A
//   static int8_t im2col_buffer[MAX_IM2COL_BUFFER_SIZE]; // Buffer B
//   static int32_t result_buffer[MAX_RESULT_BUFFER_SIZE]; // Buffer C
//   static int32_t filter_sum_per_row[4096]; 

//   int M = output_depth;
//   int K = filter_height * filter_width * input_depth;
//   int N = output_height * output_width;

//   // if ((M * K) > MAX_KERNEL_BUFFER_SIZE || (K * N) > MAX_IM2COL_BUFFER_SIZE || (M * N) > MAX_RESULT_BUFFER_SIZE) {
//   //     return; 
//   // }

//   // =================================================================
//   // STEP 1: Construct Kernel Matrix (Buffer A)
//   // =================================================================
//   for (int m = 0; m < M; ++m) {
//       int32_t current_filter_sum = 0;
//       for (int fy = 0; fy < filter_height; ++fy) {
//           for (int fx = 0; fx < filter_width; ++fx) {
//               for (int ic = 0; ic < input_depth; ++ic) {
//                   int k = fy * filter_width * input_depth + fx * input_depth + ic;
//                   int8_t f_val = filter_data[Offset(filter_shape, m, fy, fx, ic)];
//                   filter_buffer[m * K + k] = f_val;
//                   current_filter_sum += f_val;
//               }
//           }
//       }
//       filter_sum_per_row[m] = current_filter_sum;
//   }

//   // =================================================================
//   // STEP 2: Construct im2col Matrix (Buffer B)
//   // =================================================================
//   int patch_idx = 0; 
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
          
//           const int in_y_origin = (out_y * stride_height) - pad_height;
//           const int in_x_origin = (out_x * stride_width) - pad_width;
          
//           int k_idx = 0;
//           for (int fy = 0; fy < filter_height; ++fy) {
//               const int in_y = in_y_origin + dilation_height_factor * fy;
//               for (int fx = 0; fx < filter_width; ++fx) {
//                   const int in_x = in_x_origin + dilation_width_factor * fx;
                  
//                   for (int ic = 0; ic < input_depth; ++ic) {
                      
//                       bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && 
//                                                    (in_y >= 0) && (in_y < input_height);
                      
//                       int8_t val;
//                       if (is_point_inside_image) {
//                           val = input_data[Offset(input_shape, 0, in_y, in_x, ic)];
//                       } else {
//                           // FIX: Padding should be Zero Point (-input_offset)
//                           // This ensures (val + input_offset) == 0 in the math model
//                           val = static_cast<int8_t>(-input_offset); 
//                       }
                      
//                       im2col_buffer[k_idx * N + patch_idx] = val;
//                       k_idx++;
//                   }
//               }
//           }
//           patch_idx++;
//       }
//   }

//   // =================================================================
//   // STEP 3: GEMM
//   // =================================================================
//   for (int m = 0; m < M; ++m) {
//       for (int n = 0; n < N; ++n) {
//           int32_t sum = 0;
//           for (int k = 0; k < K; ++k) {
//               printf("filter = %d, im2col = %d\n", filter_buffer[m * K + k], im2col_buffer[k * N + n]);
//               sum += filter_buffer[m * K + k] * im2col_buffer[k * N + n];
//           }
//           result_buffer[m * N + n] = sum;
//       }
//   }

//   // =================================================================
//   // STEP 4: Post-processing
//   // =================================================================
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//           for (int m = 0; m < output_depth; ++m) {
//               int n = out_y * output_width + out_x;
//               int32_t acc = result_buffer[m * N + n];

//               // Add offset correction
//               acc += filter_sum_per_row[m] * input_offset;

//               if (bias_data) {
//                   acc += bias_data[m];
//               }

//               acc = MultiplyByQuantizedMultiplier(
//                   acc, output_multiplier[m], output_shift[m]);
              
//               acc += output_offset;
//               acc = std::max(acc, output_activation_min);
//               acc = std::min(acc, output_activation_max);
              
//               output_data[Offset(output_shape, 0, out_y, out_x, m)] = static_cast<int8_t>(acc);
//           }
//       }
//   }

//   perf_disable_counter(6);
// }

// -----------------------------------------------------------------------
// STATIC BUFFERS [cite: 348, 349]
// -----------------------------------------------------------------------
// #define MAX_KERNEL_BUFFER_SIZE (2048 * 512)
// #define MAX_IM2COL_BUFFER_SIZE (2048 * 512)
// #define MAX_RESULT_BUFFER_SIZE (2048 * 512)

// static int8_t filter_buffer[MAX_KERNEL_BUFFER_SIZE]; // Buffer A
// static int8_t im2col_buffer[MAX_IM2COL_BUFFER_SIZE]; // Buffer B
// static int32_t result_buffer[MAX_RESULT_BUFFER_SIZE]; // Buffer C
// static int32_t filter_sum_per_row[4096]; 

// // Fixed-point per-channel-quantization convolution reference kernel.
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
  
//   // Start Performance Counter 6 [cite: 407, 411]
//   perf_enable_counter(6);

//   // Get parameters.
//   const int32_t input_offset = params.input_offset;
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;  

//   // Set min and max value of the output.
//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency check.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);

//   int M = output_depth;
//   int K = filter_height * filter_width * input_depth;
//   int N = output_height * output_width;

//   // Buffer overflow protection
//   // if ((M * K) > MAX_KERNEL_BUFFER_SIZE || (K * N) > MAX_IM2COL_BUFFER_SIZE || (M * N) > MAX_RESULT_BUFFER_SIZE) {
//   //     perf_disable_counter(6);
//   //     return; 
//   // }

//   // =================================================================
//   // STEP 1: Construct Kernel Matrix (Buffer A) [cite: 329, 335]
//   // =================================================================
//   perf_enable_counter(0);
//   for (int m = 0; m < M; ++m) {
//       int32_t current_filter_sum = 0;
//       for (int fy = 0; fy < filter_height; ++fy) {
//           for (int fx = 0; fx < filter_width; ++fx) {
//               for (int ic = 0; ic < input_depth; ++ic) {
//                   int k = fy * filter_width * input_depth + fx * input_depth + ic;
//                   int8_t f_val = filter_data[Offset(filter_shape, m, fy, fx, ic)];
//                   filter_buffer[m * K + k] = f_val;
//                   current_filter_sum += f_val;
//               }
//           }
//       }
//       filter_sum_per_row[m] = current_filter_sum;
//   }
//   perf_disable_counter(0);
//   // =================================================================
//   // STEP 2: Construct im2col Matrix (Buffer B) [cite: 321, 323]
//   // =================================================================
//   perf_enable_counter(1);
//   int patch_idx = 0; 
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
          
//           const int in_y_origin = (out_y * stride_height) - pad_height;
//           const int in_x_origin = (out_x * stride_width) - pad_width;
          
//           int k_idx = 0;
//           for (int fy = 0; fy < filter_height; ++fy) {
//               const int in_y = in_y_origin + dilation_height_factor * fy;
//               for (int fx = 0; fx < filter_width; ++fx) {
//                   const int in_x = in_x_origin + dilation_width_factor * fx;
                  
//                   for (int ic = 0; ic < input_depth; ++ic) {
                      
//                       bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && 
//                                                    (in_y >= 0) && (in_y < input_height);
                      
//                       int8_t val;
//                       if (is_point_inside_image) {
//                           val = input_data[Offset(input_shape, 0, in_y, in_x, ic)];
//                       } else {
//                           // Correct Padding: Use Zero Point (-input_offset)
//                           val = static_cast<int8_t>(-input_offset); 
//                       }
                    
//                       im2col_buffer[k_idx * N + patch_idx] = val;
//                       k_idx++;
//                   }
//               }
//           }
//           patch_idx++;
//       }
//   }
//   perf_disable_counter(1);
//   // =================================================================
//   // STEP 3: Tiled GEMM using CFU (Corrected Data Layout)
//   // =================================================================
//   perf_enable_counter(2);
//   const int TILE_SIZE = 16; 
  
//   // Initialize result buffer to 0 for Software Accumulation
//   std::fill(result_buffer, result_buffer + M * N, 0);

//   for (int m = 0; m < M; m += TILE_SIZE) {
//       for (int n = 0; n < N; n += TILE_SIZE) {
//           for (int k = 0; k < K; k += TILE_SIZE) {
              
//               // --- 1. Send Tile A (Filter) ---
//               // Layout: Vertical Strips. Iterate Columns fast, Row-Bands slow.
//               // i goes 0..15 (Band 0, Cols 0..15), then 16..31 (Band 1, Cols 0..15)...
//               for (int i = 0; i < 64; i++) {
//                   int row_base = (i / 16) * 4; // 0, 4, 8, 12
//                   int col = i % 16;            // 0..15
//                   uint32_t packed_val = 0;
                  
//                   // Pack 4 vertical elements (Rows)
//                   for (int b = 0; b < 4; b++) {
//                       int8_t val = 0;
//                       // Boundary Check
//                       if (m + row_base + b < M && k + col < K) {
//                           val = filter_buffer[(m + row_base + b) * K + (k + col)];
//                       }
//                       // Big Endian Packing
//                       packed_val |= ((uint32_t)(val & 0xFF)) << (24 - 8 * b);
//                   }
//                   cfu_op0(1, i, packed_val);
//               }

//               // --- 2. Send Tile B (Im2Col) ---
//               // Layout: Horizontal Strips. Iterate Rows fast, Col-Bands slow.
//               // i goes 0..15 (Band 0, Rows 0..15), then 16..31 (Band 1, Rows 0..15)...
//               for (int i = 0; i < 64; i++) {
//                   int col_base = (i / 16) * 4; // 0, 4, 8, 12
//                   int row = i % 16;            // 0..15
//                   uint32_t packed_val = 0;

//                   // Pack 4 horizontal elements (Cols)
//                   for (int b = 0; b < 4; b++) {
//                       int8_t val = 0;
//                       // Boundary Check
//                       if (k + row < K && n + col_base + b < N) {
//                           val = im2col_buffer[(k + row) * N + (n + col_base + b)];
//                       }
//                       packed_val |= ((uint32_t)(val & 0xFF)) << (24 - 8 * b);
//                   }
//                   cfu_op0(2, i, packed_val);
//               }

//               // --- 3. Execute Computation ---
//               cfu_op0(3, 0, 0); // CALC

//               // --- 4. Read & Accumulate (Software Accumulation) ---
//               int C_index = 0;
//               for (int j = 0; j < 16; j += 4) {
//                   for (int i = 0; i < 16; i++) {
                      
//                       if (m + i < M && n + j < N) {
//                           result_buffer[(m + i) * N + (n + j)] += cfu_op0(4, C_index, 0);
//                       }
//                       if (m + i < M && n + j + 1 < N) {
//                           result_buffer[(m + i) * N + (n + j + 1)] += cfu_op0(4, C_index, 1);
//                       }
//                       if (m + i < M && n + j + 2 < N) {
//                           result_buffer[(m + i) * N + (n + j + 2)] += cfu_op0(4, C_index, 2);
//                       }
//                       if (m + i < M && n + j + 3 < N) {
//                           result_buffer[(m + i) * N + (n + j + 3)] += cfu_op0(4, C_index, 3);
//                       }
//                       C_index++;
//                   }
//               }

//           } // End k
//       } // End n
//   } // End m
//   perf_disable_counter(2);
//   // =================================================================
//   // STEP 4: Post-processing [cite: 356, 405]
//   // =================================================================
//   perf_enable_counter(3);
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//           for (int m = 0; m < output_depth; ++m) {
//               int n = out_y * output_width + out_x;
//               int32_t acc = result_buffer[m * N + n];

//               // Add offset correction: acc += (Sum of weights) * input_offset
//               acc += filter_sum_per_row[m] * input_offset;

//               if (bias_data) {
//                   acc += bias_data[m];
//               }

//               acc = MultiplyByQuantizedMultiplier(
//                   acc, output_multiplier[m], output_shift[m]);
              
//               acc += output_offset;
//               acc = std::max(acc, output_activation_min);
//               acc = std::min(acc, output_activation_max);
              
//               output_data[Offset(output_shape, 0, out_y, out_x, m)] = static_cast<int8_t>(acc);
//           }
//       }
//   }
//   perf_disable_counter(3);
//   // Stop Performance Counter 6 [cite: 413]
//   perf_disable_counter(6);
// }

// Fixed-point per-channel-quantization convolution reference kernel.
// inline void ConvPerChannel(
//     const ConvParams& params, const int32_t* output_multiplier,
//     const int32_t* output_shift, const RuntimeShape& input_shape,
//     const int8_t* input_data, const RuntimeShape& filter_shape,
//     const int8_t* filter_data, const RuntimeShape& bias_shape,
//     const int32_t* bias_data, const RuntimeShape& output_shape,
//     int8_t* output_data) {
  
//   // Start Performance Counter 6 [cite: 407, 411]
//   perf_enable_counter(6);

//   // Get parameters.
//   const int32_t input_offset = params.input_offset;
//   const int stride_width = params.stride_width;
//   const int stride_height = params.stride_height;
//   const int dilation_width_factor = params.dilation_width_factor;
//   const int dilation_height_factor = params.dilation_height_factor;
//   const int pad_width = params.padding_values.width;
//   const int pad_height = params.padding_values.height;
//   const int32_t output_offset = params.output_offset;

//   const int32_t output_activation_min = params.quantized_activation_min;
//   const int32_t output_activation_max = params.quantized_activation_max;

//   // Consistency check.
//   TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
//   TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
//   TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  
//   const int input_depth = input_shape.Dims(3);
//   const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  
//   if (bias_data) {
//     TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
//   }

//   const int input_height = input_shape.Dims(1);
//   const int input_width = input_shape.Dims(2);
//   const int filter_height = filter_shape.Dims(1);
//   const int filter_width = filter_shape.Dims(2);
//   const int output_height = output_shape.Dims(1);
//   const int output_width = output_shape.Dims(2);

//   int M = output_depth;
//   int K = filter_height * filter_width * input_depth;
//   int N = output_height * output_width;

//   // -----------------------------------------------------------------------
//   // STATIC BUFFERS [cite: 348, 349]
//   // -----------------------------------------------------------------------
//   #define MAX_KERNEL_BUFFER_SIZE (2048 * 512)
//   #define MAX_IM2COL_BUFFER_SIZE (2048 * 512)
//   #define MAX_RESULT_BUFFER_SIZE (2048 * 512)
  
//   // Pre-packed buffers (uint32_t) size approx 1/4 of int8 + safety padding
//   #define MAX_PACKED_SIZE (MAX_KERNEL_BUFFER_SIZE / 4 + 1024)

//   static int8_t filter_buffer[MAX_KERNEL_BUFFER_SIZE]; 
//   static int8_t im2col_buffer[MAX_IM2COL_BUFFER_SIZE]; 
//   static int32_t result_buffer[MAX_RESULT_BUFFER_SIZE]; 
//   static int32_t filter_sum_per_row[4096]; 

//   // [OPTIMIZATION] Pre-packed buffers
//   static uint32_t packed_filter_buffer[MAX_PACKED_SIZE];
//   static uint32_t packed_im2col_buffer[MAX_PACKED_SIZE];

//   // =================================================================
//   // STEP 1: Construct Kernel Matrix (Buffer A) [cite: 329, 335]
//   // =================================================================
//   for (int m = 0; m < M; ++m) {
//       int32_t current_filter_sum = 0;
//       for (int fy = 0; fy < filter_height; ++fy) {
//           for (int fx = 0; fx < filter_width; ++fx) {
//               for (int ic = 0; ic < input_depth; ++ic) {
//                   int k = fy * filter_width * input_depth + fx * input_depth + ic;
//                   int8_t f_val = filter_data[Offset(filter_shape, m, fy, fx, ic)];
//                   filter_buffer[m * K + k] = f_val;
//                   current_filter_sum += f_val;
//               }
//           }
//       }
//       filter_sum_per_row[m] = current_filter_sum;
//   }

//   // [OPTIMIZATION] Pre-pack Filter Buffer (Buffer A)
//   // Layout: Block Row (M) -> Block Col (K) -> Tile Data (Vertical Strips)
//   int packed_A_idx = 0;
//   for (int m = 0; m < M; m += 16) {
//       for (int k = 0; k < K; k += 16) {
//           // Pack 16x16 tile
//           for (int i = 0; i < 64; i++) {
//               int row_base = (i / 16) * 4; 
//               int col = i % 16;            
//               uint32_t packed_val = 0;
//               for (int b = 0; b < 4; b++) {
//                   int8_t val = 0;
//                   if (m + row_base + b < M && k + col < K) {
//                       val = filter_buffer[(m + row_base + b) * K + (k + col)];
//                   }
//                   // Big Endian Packing
//                   packed_val |= ((uint32_t)(val & 0xFF)) << (24 - 8 * b);
//               }
//               packed_filter_buffer[packed_A_idx++] = packed_val;
//           }
//       }
//   }

//   // =================================================================
//   // STEP 2: Construct im2col Matrix (Buffer B) [cite: 321, 323]
//   // =================================================================
//   int patch_idx = 0; 
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//           const int in_y_origin = (out_y * stride_height) - pad_height;
//           const int in_x_origin = (out_x * stride_width) - pad_width;
//           int k_idx = 0;
//           for (int fy = 0; fy < filter_height; ++fy) {
//               const int in_y = in_y_origin + dilation_height_factor * fy;
//               for (int fx = 0; fx < filter_width; ++fx) {
//                   const int in_x = in_x_origin + dilation_width_factor * fx;
//                   for (int ic = 0; ic < input_depth; ++ic) {
//                       bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && 
//                                                    (in_y >= 0) && (in_y < input_height);
//                       int8_t val;
//                       if (is_point_inside_image) {
//                           val = input_data[Offset(input_shape, 0, in_y, in_x, ic)];
//                       } else {
//                           // Correct Padding: Use Zero Point (-input_offset)
//                           val = static_cast<int8_t>(-input_offset); 
//                       }
//                       im2col_buffer[k_idx * N + patch_idx] = val;
//                       k_idx++;
//                   }
//               }
//           }
//           patch_idx++;
//       }
//   }

//   // [OPTIMIZATION] Pre-pack Im2Col Buffer (Buffer B)
//   // Layout: Block Row (N) -> Block Col (K) -> Tile Data (Horizontal Strips)
//   int packed_B_idx = 0;
//   for (int n = 0; n < N; n += 16) {
//       for (int k = 0; k < K; k += 16) {
//           // Pack 16x16 tile
//           for (int i = 0; i < 64; i++) {
//               int col_base = (i / 16) * 4; 
//               int row = i % 16;            
//               uint32_t packed_val = 0;
//               for (int b = 0; b < 4; b++) {
//                   int8_t val = 0;
//                   if (k + row < K && n + col_base + b < N) {
//                       val = im2col_buffer[(k + row) * N + (n + col_base + b)];
//                   }
//                   packed_val |= ((uint32_t)(val & 0xFF)) << (24 - 8 * b);
//               }
//               packed_im2col_buffer[packed_B_idx++] = packed_val;
//           }
//       }
//   }

//   // =================================================================
//   // STEP 3: Tiled GEMM using CFU (Optimized with Pre-packing + Correct Accumulation) [cite: 23, 388, 393]
//   // =================================================================
//   const int TILE_SIZE = 16; 
//   // Initialize result buffer to 0
//   std::fill(result_buffer, result_buffer + M * N, 0);

//   // Pre-calculate block dimensions for indexing
//   int K_blocks = (K + 15) / 16;

//   // Base pointers
//   const uint32_t* ptr_A_base = packed_filter_buffer;
//   const uint32_t* ptr_B_base = packed_im2col_buffer;

//   for (int m = 0; m < M; m += TILE_SIZE) {
//       int m_blk = m / 16;

//       for (int n = 0; n < N; n += TILE_SIZE) {
//           int n_blk = n / 16;

//           for (int k = 0; k < K; k += TILE_SIZE) {
//               int k_blk = k / 16;

//               // --- Calculate Pointers ---
//               // A: [m_blk][k_blk]
//               const uint32_t* ptr_A = ptr_A_base + (m_blk * K_blocks + k_blk) * 64;
//               // B: [n_blk][k_blk]
//               const uint32_t* ptr_B = ptr_B_base + (n_blk * K_blocks + k_blk) * 64;

//               // --- 1. Send Tile A (Filter) ---
//               for (int i = 0; i < 64; i++) {
//                   cfu_op0(1, i, ptr_A[i]);
//               }

//               // --- 2. Send Tile B (Im2Col) ---
//               for (int i = 0; i < 64; i++) {
//                   cfu_op0(2, i, ptr_B[i]);
//               }

//               // --- 3. Execute Computation ---
//               cfu_op0(3, 0, 0); // CALC

//               // --- 4. Read & Accumulate (INSIDE K Loop) ---
//               // We MUST accumulate here because HW is stateless (no reset used)
//               int C_index = 0;
//               for (int j = 0; j < 16; j += 4) {
//                   for (int i = 0; i < 16; i++) {
                      
//                       if (m + i < M && n + j < N) {
//                           result_buffer[(m + i) * N + (n + j)] += cfu_op0(4, C_index, 0);
//                       }
//                       if (m + i < M && n + j + 1 < N) {
//                           result_buffer[(m + i) * N + (n + j + 1)] += cfu_op0(4, C_index, 1);
//                       }
//                       if (m + i < M && n + j + 2 < N) {
//                           result_buffer[(m + i) * N + (n + j + 2)] += cfu_op0(4, C_index, 2);
//                       }
//                       if (m + i < M && n + j + 3 < N) {
//                           result_buffer[(m + i) * N + (n + j + 3)] += cfu_op0(4, C_index, 3);
//                       }
//                       C_index++;
//                   }
//               }
//           } // End k
//       } // End n
//   } // End m

//   // =================================================================
//   // STEP 4: Post-processing [cite: 356, 405]
//   // =================================================================
//   for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//           for (int m = 0; m < output_depth; ++m) {
//               int n = out_y * output_width + out_x;
//               int32_t acc = result_buffer[m * N + n];

//               // Add offset correction
//               acc += filter_sum_per_row[m] * input_offset;

//               if (bias_data) {
//                   acc += bias_data[m];
//               }

//               acc = MultiplyByQuantizedMultiplier(
//                   acc, output_multiplier[m], output_shift[m]);
              
//               acc += output_offset;
//               acc = std::max(acc, output_activation_min);
//               acc = std::min(acc, output_activation_max);
              
//               output_data[Offset(output_shape, 0, out_y, out_x, m)] = static_cast<int8_t>(acc);
//           }
//       }
//   }

//   // Stop Performance Counter 6 [cite: 413]
//   perf_disable_counter(6);
// }

// version3


// -----------------------------------------------------------------------
// STATIC BUFFERS
// -----------------------------------------------------------------------
#define MAX_KERNEL_BUFFER_SIZE (4096 * 4096) 
#define MAX_IM2COL_BUFFER_SIZE (2048 * 2048)
#define MAX_RESULT_BUFFER_SIZE (2048 * 2048)
#define MAX_PACKED_SIZE (MAX_KERNEL_BUFFER_SIZE / 4 + 4096)

static int8_t filter_buffer[MAX_KERNEL_BUFFER_SIZE]; 
static int8_t im2col_buffer[MAX_IM2COL_BUFFER_SIZE]; 
static int32_t result_buffer[MAX_RESULT_BUFFER_SIZE]; 
static int32_t filter_sum_per_row[4096]; 

// [OPTIMIZATION] Pre-packed buffers
static uint32_t packed_filter_buffer[MAX_PACKED_SIZE];
static uint32_t packed_im2col_buffer[MAX_PACKED_SIZE];   


inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  // Get parameters.
  const int32_t input_offset = params.input_offset;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  const int32_t output_activation_min = -128; // params.quantized_activation_min;
  const int32_t output_activation_max = 127; // params.quantized_activation_max;

//   print_conv_params(params, input_shape, filter_shape, output_shape); // print parameter to fix it

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int M = output_depth;
  int K = filter_height * filter_width * input_depth;
  int N = output_height * output_width;

  // =================================================================
  // STEP 1: Construct Kernel Matrix (Buffer A)
  // =================================================================
  for (int m = 0; m < M; ++m) {
      int32_t current_filter_sum = 0;
      for (int fy = 0; fy < filter_height; ++fy) {
          for (int fx = 0; fx < filter_width; ++fx) {
              for (int ic = 0; ic < input_depth; ++ic) {
                  int k = fy * filter_width * input_depth + fx * input_depth + ic;
                  int8_t f_val = filter_data[Offset(filter_shape, m, fy, fx, ic)];
                  filter_buffer[m * K + k] = f_val;
                  current_filter_sum += f_val;
              }
          }
      }
      filter_sum_per_row[m] = current_filter_sum;
  }

  // [OPTIMIZATION] Pre-pack Filter Buffer (Buffer A)
  int packed_A_idx = 0;
  for (int m = 0; m < M; m += 16) {
      for (int k = 0; k < K; k += 16) {
          for (int i = 0; i < 64; i++) {
              int row_base = (i / 16) * 4; 
              int col = i % 16;            
              uint32_t packed_val = 0;
              // Loop unrolling for packing
              int base_idx = (m + row_base) * K + (k + col);
              
              // Safe check omitted for brevity in optimized loop, assume padding handles it or check bounds
              // Optimistic packing assuming M and K are multiples of 16 or buffers are padded
              int8_t val0 = (m + row_base + 0 < M && k + col < K) ? filter_buffer[base_idx] : 0;
              int8_t val1 = (m + row_base + 1 < M && k + col < K) ? filter_buffer[base_idx + K] : 0;
              int8_t val2 = (m + row_base + 2 < M && k + col < K) ? filter_buffer[base_idx + 2*K] : 0;
              int8_t val3 = (m + row_base + 3 < M && k + col < K) ? filter_buffer[base_idx + 3*K] : 0;

              packed_val = ((uint32_t)(val0 & 0xFF) << 24) |
                           ((uint32_t)(val1 & 0xFF) << 16) |
                           ((uint32_t)(val2 & 0xFF) << 8)  |
                           ((uint32_t)(val3 & 0xFF));
              
              packed_filter_buffer[packed_A_idx++] = packed_val;
          }
      }
  }

  // =================================================================
  // STEP 2: Construct im2col Matrix (Buffer B)
  // =================================================================
  int patch_idx = 0; 
  for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_y_origin = (out_y * stride_height) - pad_height;
          const int in_x_origin = (out_x * stride_width) - pad_width;
          int k_idx = 0;
          for (int fy = 0; fy < filter_height; ++fy) {
              const int in_y = in_y_origin + dilation_height_factor * fy;
              for (int fx = 0; fx < filter_width; ++fx) {
                  const int in_x = in_x_origin + dilation_width_factor * fx;
                  for (int ic = 0; ic < input_depth; ++ic) {
                      bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && 
                                                   (in_y >= 0) && (in_y < input_height);
                      int8_t val;
                      if (is_point_inside_image) {
                          val = input_data[Offset(input_shape, 0, in_y, in_x, ic)];
                      } else {
                          // Correct Padding: Use Zero Point (-input_offset)
                          val = static_cast<int8_t>(-input_offset); 
                      }
                      im2col_buffer[k_idx * N + patch_idx] = val;
                      k_idx++;
                  }
              }
          }
          patch_idx++;
      }
  }

  // [OPTIMIZATION] Pre-pack Im2Col Buffer (Buffer B)
  int packed_B_idx = 0;
  for (int n = 0; n < N; n += 16) {
      for (int k = 0; k < K; k += 16) {
          for (int i = 0; i < 64; i++) {
              int col_base = (i / 16) * 4; 
              int row = i % 16;            
              
              int base_idx = (k + row) * N + (n + col_base);
              
              int8_t val0 = (k + row < K && n + col_base + 0 < N) ? im2col_buffer[base_idx] : 0;
              int8_t val1 = (k + row < K && n + col_base + 1 < N) ? im2col_buffer[base_idx + 1] : 0;
              int8_t val2 = (k + row < K && n + col_base + 2 < N) ? im2col_buffer[base_idx + 2] : 0;
              int8_t val3 = (k + row < K && n + col_base + 3 < N) ? im2col_buffer[base_idx + 3] : 0;

              uint32_t packed_val = ((uint32_t)(val0 & 0xFF) << 24) |
                                    ((uint32_t)(val1 & 0xFF) << 16) |
                                    ((uint32_t)(val2 & 0xFF) << 8)  |
                                    ((uint32_t)(val3 & 0xFF));
              packed_im2col_buffer[packed_B_idx++] = packed_val;
          }
      }
  }


  // =================================================================
  // STEP 3: Tiled GEMM using CFU 
  // =================================================================
  const int TILE_SIZE = 16; 
  std::fill(result_buffer, result_buffer + M * N, 0);

  int K_blocks = (K + 15) / 16;

  const uint32_t* ptr_A_base = packed_filter_buffer;
  const uint32_t* ptr_B_base = packed_im2col_buffer;

  for (int m = 0; m < M; m += TILE_SIZE) {
      int m_blk = m / 16;

      for (int n = 0; n < N; n += TILE_SIZE) {
          int n_blk = n / 16;

          for (int k = 0; k < K; k += TILE_SIZE) {
              int k_blk = k / 16;

              // --- Calculate Pointers ---
              const uint32_t* ptr_A = ptr_A_base + (m_blk * K_blocks + k_blk) * 64;
              const uint32_t* ptr_B = ptr_B_base + (n_blk * K_blocks + k_blk) * 64;

              // --- 1. Send Tile A (Filter) --- [TPU bigger]
              for (int i = 0; i < 64; i++) {
                printf("i = %d\n", i);
                //   cfu_op0(7, ptr_A[i], ptr_B[i]);
                  cfu_op0(1, i, ptr_A[i]);
              }

              // --- 2. Send Tile B (Im2Col) ---
              for (int i = 0; i < 64; i++) {
                  cfu_op0(2, i, ptr_B[i]);
              }

              // --- 3. Execute Computation ---
              cfu_op0(3, 0, 0); // CALC

              // --- 4. Read & Accumulate ---
              int C_index = 0;
              for (int j = 0; j < 16; j += 4) {
                  for (int i = 0; i < 16; i++) {
                      
                      // 讀取 CFU 計算結果 (一次讀回 4 個部分結果)
                      int32_t res0 = cfu_op0(4, C_index, 0);
                      int32_t res1 = cfu_op0(4, C_index, 1);
                      int32_t res2 = cfu_op0(4, C_index, 2);
                      int32_t res3 = cfu_op0(4, C_index, 3);
                      
                      // 這裡的邊界檢查 (m+i < M) 其實可以提到迴圈外或利用 Padding 簡化，
                      // 但為了保持正確性先維持原樣。
                      if (m + i < M) { 
                          int row_idx = (m + i) * N;
                          if (n + j < N)     result_buffer[row_idx + (n + j)]     += res0;
                          if (n + j + 1 < N) result_buffer[row_idx + (n + j + 1)] += res1;
                          if (n + j + 2 < N) result_buffer[row_idx + (n + j + 2)] += res2;
                          if (n + j + 3 < N) result_buffer[row_idx + (n + j + 3)] += res3;
                      }
                      C_index++;
                  }
              }
          } // End k
      } // End n
  } // End m

  // =================================================================
  // STEP 4: Post-processing
  // =================================================================
  for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int m = 0; m < output_depth; ++m) {
              int n = out_y * output_width + out_x;
              int32_t acc = result_buffer[m * N + n];

              // Add offset correction 
              acc += filter_sum_per_row[m] * input_offset;

              if (bias_data) {
                  acc += bias_data[m];
              }

              acc = MultiplyByQuantizedMultiplier(
                  acc, output_multiplier[m], output_shift[m]);
              
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              
              output_data[Offset(output_shape, 0, out_y, out_x, m)] = static_cast<int8_t>(acc);
          }
      }
  }
}




inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

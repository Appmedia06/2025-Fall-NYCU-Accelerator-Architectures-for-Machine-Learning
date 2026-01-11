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
#include <vector>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "cfu.h"
#include <cstdio>
#include "playground_util/print_params.h"

namespace tflite {
namespace reference_integer_ops {

// -----------------------------------------------------------------------
// STATIC BUFFERS
// -----------------------------------------------------------------------
#define MAX_KERNEL_BUFFER_SIZE (4096 * 4096) 
#define MAX_IM2COL_BUFFER_SIZE (2048 * 2048)
#define MAX_RESULT_BUFFER_SIZE (2048 * 2048)
#define MAX_PACKED_SIZE (MAX_KERNEL_BUFFER_SIZE / 4 + 8192)
#define MAX_OUTPUT_DEPTH 4096

alignas(64) static int8_t filter_buffer[MAX_KERNEL_BUFFER_SIZE]; 
alignas(64) static int8_t im2col_buffer[MAX_IM2COL_BUFFER_SIZE]; 
alignas(64) static int32_t result_buffer[MAX_RESULT_BUFFER_SIZE]; 
alignas(64) static int32_t filter_sum_per_row[4096]; 
alignas(64) static int32_t effective_bias[MAX_OUTPUT_DEPTH];

// [OPTIMIZATION] Pre-packed buffers
alignas(64) static uint32_t packed_filter_buffer[MAX_PACKED_SIZE];
alignas(64) static uint32_t packed_im2col_buffer[MAX_PACKED_SIZE];    

// ***changed chriceu***
static const int8_t* cached_filter_ptr = nullptr;
static int cached_M = -1;
static int cached_K = -1;
static bool filter_packed_valid = false;


// ***changed

inline int32_t my_MultiplyByQuantizedMultiplier(int32_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {                                           
  cfu_op0(8, x, 0);
  return cfu_op0(9, quantized_multiplier, shift);

}

inline int my_Offset(const RuntimeShape& shape, int i0, int i1, int i2, int i3) {
  const int* dims_data = reinterpret_cast<const int*>(shape.DimsData());
  return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
}



inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  const int32_t input_offset = params.input_offset;
  const int stride_width = params.stride_width;
  const int stride_height = 1; 
  const int dilation_width_factor = 1; 
  const int dilation_height_factor = 1; 
  const int pad_width = params.padding_values.width;
  const int pad_height = 0; 
  const int32_t output_offset = params.output_offset;

  const int32_t output_activation_min = -128; 
  const int32_t output_activation_max = 127; 

  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // 定義新的 Tile 尺寸 (已修改)
  // Matrix A (Filter): TILE_M x TILE_K = 252 x 512
  // Matrix B (Input):  TILE_K x TILE_N = 512 x 148
  // Output:            TILE_M x TILE_N = 252 x 148
  const int TILE_M = 252; // 修改這裡：從 250 改為 252
  const int TILE_K = 624;
  const int TILE_N = 148;

  // 每個 Tile 包含多少個 32-bit words (每個 word 包含 4 個 int8)
  // 計算：(252 * 512) / 4 = 32256 words
  const int WORDS_PER_TILE_A = (TILE_M * TILE_K) / 4; 
  // 計算：(512 * 148) / 4 = 18944 words (維持不變)
  const int WORDS_PER_TILE_B = (TILE_K * TILE_N) / 4; 

  int M = output_depth;
  int K = filter_height * filter_width * input_depth;
  int N = output_height * output_width; 

    // =================================================================
    // STEP 1: Construct Kernel Matrix (Buffer A) - Flattening & Packing Optimized
    // =================================================================
    const int8_t* src_ptr = filter_data;
    int8_t* dst_ptr = filter_buffer;

    // 檢查 Cache 是否有效，避免重複執行
    if (!filter_packed_valid ||
        cached_filter_ptr != filter_data ||
        cached_M != M ||
        cached_K != K) [[unlikely]] {

        // ---------------------------------------------------------
        // 1. Flatten Filter Data (由 strided 轉為 linear)
        // ---------------------------------------------------------
        // 優化：使用指針遞增 (*src_ptr++) 取代 my_Offset 計算
        for (int m = 0; m < M; ++m) {
            int32_t current_filter_sum = 0;
            for (int fy = 0; fy < filter_height; ++fy) {
                for (int fx = 0; fx < filter_width; ++fx) {
                    for (int ic = 0; ic < input_depth; ++ic) {
                        int8_t f_val = *src_ptr++;  // 線性讀取來源
                        *dst_ptr++ = f_val;         // 線性寫入 Buffer A
                        current_filter_sum += f_val;
                    }
                }
            }
            filter_sum_per_row[m] = current_filter_sum;
        }

        // ---------------------------------------------------------
        // 2. Pre-pack Filter Buffer (Buffer A) - 252 x 512
        // ---------------------------------------------------------
        int packed_A_idx = 0;
        
        // 定義 Tile 尺寸 (根據您第一段程式碼的邏輯)
        const int TILE_M_SIZE = 252; 
        const int TILE_K_SIZE = 624;

        // 外層迴圈：步進 252
        for (int m0 = 0; m0 < M; m0 += TILE_M_SIZE) {
            // 判斷是否為完整的一個 Tile (高度足夠)
            const bool full_m = (m0 + TILE_M_SIZE - 1 < M);

            // 內層迴圈：步進 512
            for (int k0 = 0; k0 < K; k0 += TILE_K_SIZE) {
                // 判斷是否為完整的一個 Tile (寬度足夠)
                const bool full_k = (k0 + TILE_K_SIZE - 1 < K);

                // [Fast Path] 核心區域：不需要邊界檢查
                if (full_m && full_k) [[likely]]   {
                    // 252 rows, 每次處理 4 rows
                    for (int rb = 0; rb < TILE_M_SIZE; rb += 4) {
                        // 預先計算好 4 個 row 的起始指針偏移量
                        int base0 = (m0 + rb + 0) * K + k0;
                        int base1 = base0 + K;
                        int base2 = base1 + K;
                        int base3 = base2 + K;

                        for (int col = 0; col < TILE_K_SIZE; col += 4) {
                            int base0_plus_col = base0 + col;
                            int base1_plus_col = base1 + col;
                            int base2_plus_col = base2 + col;
                            int base3_plus_col = base3 + col;

                            uint32_t v0s = *reinterpret_cast<uint32_t *>(&filter_buffer[base0_plus_col]);
                            uint32_t v1s = *reinterpret_cast<uint32_t *>(&filter_buffer[base1_plus_col]);
                            uint32_t v2s = *reinterpret_cast<uint32_t *>(&filter_buffer[base2_plus_col]);
                            uint32_t v3s = *reinterpret_cast<uint32_t *>(&filter_buffer[base3_plus_col]);

                            packed_filter_buffer[packed_A_idx++] = 
                                ((v0s & 0x000000FF) << 24) |
                                ((v1s & 0x000000FF) << 16) |
                                ((v2s & 0x000000FF) << 8)  |
                                ((v3s & 0x000000FF));

                            packed_filter_buffer[packed_A_idx++] = 
                                ((v0s & 0x0000FF00) << 16) |
                                ((v1s & 0x0000FF00) << 8) |
                                ((v2s & 0x0000FF00)) |
                                ((v3s & 0x0000FF00) >> 8);

                            packed_filter_buffer[packed_A_idx++] = 
                                ((v0s & 0x00FF0000) << 8) |
                                ((v1s & 0x00FF0000)) |
                                ((v2s & 0x00FF0000) >> 8) |
                                ((v3s & 0x00FF0000) >> 16);

                            packed_filter_buffer[packed_A_idx++] = 
                                ((v0s & 0xFF000000)) |
                                ((v1s & 0xFF000000) >> 8) |
                                ((v2s & 0xFF000000) >> 16)  |
                                ((v3s & 0xFF000000) >> 24);
                        }
                    }
                } 
                // [Slow Path] 邊緣區域：需要邊界檢查 (Padding)
                else {
                    for (int rb = 0; rb < TILE_M_SIZE; rb += 4) {
                        for (int col = 0; col < TILE_K_SIZE; ++col) {
                            int mm0 = m0 + rb + 0;
                            int mm1 = m0 + rb + 1;
                            int mm2 = m0 + rb + 2;
                            int mm3 = m0 + rb + 3;
                            int kk  = k0 + col;

                            int8_t v0 = (mm0 < M && kk < K) ? filter_buffer[mm0 * K + kk] : 0;
                            int8_t v1 = (mm1 < M && kk < K) ? filter_buffer[mm1 * K + kk] : 0;
                            int8_t v2 = (mm2 < M && kk < K) ? filter_buffer[mm2 * K + kk] : 0;
                            int8_t v3 = (mm3 < M && kk < K) ? filter_buffer[mm3 * K + kk] : 0;

                            packed_filter_buffer[packed_A_idx++] =
                                ((uint32_t)(v0 & 0xFF) << 24) |
                                ((uint32_t)(v1 & 0xFF) << 16) |
                                ((uint32_t)(v2 & 0xFF) << 8)  |
                                ((uint32_t)(v3 & 0xFF));
                        }
                    }
                }
            }
        }

        // 更新 Cache 狀態
        cached_filter_ptr = filter_data;
        cached_M = M;
        cached_K = K;
        filter_packed_valid = true;
        // printf("packed_A_idx = %d\n", packed_A_idx);
    }
    

    // =================================================================
    // STEP 2: Construct im2col Matrix (Buffer B) - Optimized
    // =================================================================
    int patch_idx = 0;
    const int8_t pad_val = static_cast<int8_t>(-input_offset);

    // 預先計算 stride 避免迴圈內重複計算
    // const int input_stride_y = input_width * input_depth;
    // const int input_stride_x = input_depth;

    for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;

        for (int out_x = 0; out_x < output_width; ++out_x) {
            const int in_x_origin = (out_x * stride_width) - pad_width;

            // 設定寫入指標：Buffer B 是 (K x N) column-major 概念，但在記憶體是 row-major (K rows, N cols)
            // 每個 patch 對應 buffer 中的一個 column
            int8_t* dst2_ptr = im2col_buffer + patch_idx; 
            
            // 判斷是否為完整的 Patch (不需要 Padding)
            const int in_y_last = in_y_origin + (filter_height - 1) * dilation_height_factor;
            const int in_x_last = in_x_origin + (filter_width  - 1) * dilation_width_factor;

            const bool full_patch =
                (in_x_origin >= 0) && (in_y_origin >= 0) &&
                (in_x_last  < input_width) && (in_y_last < input_height);

            // ---------------------------------------------------------
            // Im2Col Generation: Fast Path vs Slow Path
            // ---------------------------------------------------------
            if (full_patch) [[likely]]  {
                // [Fast Path] 無需邊界檢查，直接指針操作
                for (int fy = 0; fy < filter_height; ++fy) {
                    const int in_y = in_y_origin + dilation_height_factor * fy;
                    // 優化：預先算出該 row 的起始指標
                    const int8_t* src_row_base = input_data + my_Offset(input_shape, 0, in_y, 0, 0);

                    for (int fx = 0; fx < filter_width; ++fx) {
                        const int in_x = in_x_origin + dilation_width_factor * fx;
                        const int8_t* src = src_row_base + (in_x * input_depth); // 快速定位

                        for (int ic = 0; ic < input_depth; ++ic) {
                            *dst2_ptr = src[ic];
                            dst2_ptr += N; // 跳到下一個 K (在 Buffer B 中 stride 為 N)
                        }
                    }
                }
            } else {
                // [Slow Path] 需要邊界檢查 (Padding)
                for (int fy = 0; fy < filter_height; ++fy) {
                    const int in_y = in_y_origin + dilation_height_factor * fy;
                    const bool y_ok = (in_y >= 0) && (in_y < input_height);
                    
                    for (int fx = 0; fx < filter_width; ++fx) {
                        const int in_x = in_x_origin + dilation_width_factor * fx;
                        const bool x_ok = (in_x >= 0) && (in_x < input_width);
                        
                        if (x_ok && y_ok) [[likely]]  {
                            const int8_t* src = input_data + my_Offset(input_shape, 0, in_y, in_x, 0);
                            for (int ic = 0; ic < input_depth; ++ic) {
                                *dst2_ptr = src[ic];
                                dst2_ptr += N;
                            }
                        } else {
                            for (int ic = 0; ic < input_depth; ++ic) {
                                *dst2_ptr = pad_val;
                                dst2_ptr += N;
                            }
                        }
                    }
                }
            }
            patch_idx++;
        }
    }

    // =================================================================
    // [OPTIMIZATION] Pre-pack Im2Col Buffer (Buffer B) - 512 x 148
    // =================================================================
    int packed_B_idx = 0;
    const int TILE_N_SIZE = 148;
    const int TILE_K_SIZE = 624;

    // [Modified] N-Loop Step: 148
    for (int n0 = 0; n0 < N; n0 += TILE_N_SIZE) {
        const bool full_n = (n0 + TILE_N_SIZE - 1 < N);

        // [Modified] K-Loop Step: 512
        for (int k0 = 0; k0 < K; k0 += TILE_K_SIZE) {
            const bool full_k = (k0 + TILE_K_SIZE - 1 < K);

            // Packing Logic (148 cols, 4 cols at a time)
            // 148 / 4 = 37 blocks (整除! 不需要 goto 或特殊處理)
            
            if (full_n && full_k) [[likely]]   {
                // [Fast Path] 核心區域
                for (int col_base = 0; col_base < TILE_N_SIZE; col_base += 4) {
                    // 這裡的邏輯是：在 im2col_buffer 中，同一 row 的不同 column 是連續的 (N 維度)
                    // 所以 val0, val1, val2, val3 是連續記憶體位置
                    int n_idx = n0 + col_base;
                    
                    for (int row = 0; row < TILE_K_SIZE; ++row) {
                        int k_idx = k0 + row;
                        int base_addr = k_idx * N + n_idx;

                        uint32_t vs = *reinterpret_cast<uint32_t *>(&im2col_buffer[base_addr]);

                        packed_im2col_buffer[packed_B_idx++] = __builtin_bswap32(vs);
                    }
                }
            } else [[unlikely]]  {
                // [Slow Path] 邊界區域
                for (int col_base = 0; col_base < TILE_N_SIZE; col_base += 4) {
                    int patch0 = n0 + col_base + 0;
                    int patch1 = n0 + col_base + 1;
                    int patch2 = n0 + col_base + 2;
                    int patch3 = n0 + col_base + 3;

                    for (int row = 0; row < TILE_K_SIZE; ++row) {
                        int kk = k0 + row;
                        int base_row_idx = kk * N; // 這一行的起始點

                        int8_t v0 = (kk < K && patch0 < N) ? im2col_buffer[base_row_idx + patch0] : 0;
                        int8_t v1 = (kk < K && patch1 < N) ? im2col_buffer[base_row_idx + patch1] : 0;
                        int8_t v2 = (kk < K && patch2 < N) ? im2col_buffer[base_row_idx + patch2] : 0;
                        int8_t v3 = (kk < K && patch3 < N) ? im2col_buffer[base_row_idx + patch3] : 0;

                        packed_im2col_buffer[packed_B_idx++] =
                            ((uint32_t)(v0 & 0xFF) << 24) |
                            ((uint32_t)(v1 & 0xFF) << 16) |
                            ((uint32_t)(v2 & 0xFF) << 8)  |
                            ((uint32_t)(v3 & 0xFF));
                    }
                }
            }
        }
    }
    // printf("packed_B_idx = %d\n", packed_B_idx);

  // =================================================================
  // STEP 3: Tiled GEMM using CFU (252x512 * 512x148 -> 252x148)
  // =================================================================
  std::fill(result_buffer, result_buffer + M * N, 0);

  // 計算 K block 的數量，以 512 為單位向上取整
  int K_blocks = (K + TILE_K - 1) / TILE_K;

  const uint32_t* ptr_A_base = packed_filter_buffer;
  const uint32_t* ptr_B_base = packed_im2col_buffer;

//   int cnt = 0;

  // Loop M (row) step 252
  for (int m = 0; m < M; m += TILE_M) {
      int m_blk = m / TILE_M;

      // Loop N (col) step 148
      for (int n = 0; n < N; n += TILE_N) {
          int n_blk = n / TILE_N;

          // Loop K (depth) step 512
          for (int k = 0; k < K; k += TILE_K) {

              int k_blk = k / TILE_K;

            //   cnt++;

              // 指標計算：根據新的 words 數量跳轉
              const uint32_t* ptr_A = ptr_A_base + (m_blk * K_blocks + k_blk) * WORDS_PER_TILE_A;
              const uint32_t* ptr_B = ptr_B_base + (n_blk * K_blocks + k_blk) * WORDS_PER_TILE_B;

              // 1. Send Tile A (252x512 -> 32256 words)
              // 注意：迴圈長度會自動根據 WORDS_PER_TILE_A 更新
              for (int i = 0; i < WORDS_PER_TILE_A; i+=8) {
                 cfu_op0(1, ptr_A[i],   ptr_A[i+1]); 
                 cfu_op0(1, ptr_A[i+2], ptr_A[i+3]); 
                 cfu_op0(1, ptr_A[i+4], ptr_A[i+5]); 
                 cfu_op0(1, ptr_A[i+6], ptr_A[i+7]); 
              }

              // 2. Send Tile B (512x148 -> 18944 words)
              for (int i = 0; i < WORDS_PER_TILE_B; i+=8) {
                 cfu_op0(2, ptr_B[i],   ptr_B[i+1]); 
                 cfu_op0(2, ptr_B[i+2], ptr_B[i+3]); 
                 cfu_op0(2, ptr_B[i+4], ptr_B[i+5]); 
                 cfu_op0(2, ptr_B[i+6], ptr_B[i+7]); 
              }     

              // 3. Execute
            //   cfu_op0(11, 0, 0); 

              // 4. Read & Accumulate (Result Tile size: 252x148)
              // -----------------------------------------------------------
              int C_index = 0;
              // Loop Columns (N dimension) -> 148
              for (int j = 0; j < TILE_N; j += 4) { 
                  // Loop Rows (M dimension) -> 252 (這裡會自動變成 252)
                  for (int i = 0; i < TILE_M; i++) { 
                      
                      int32_t res0 = cfu_op0(4, C_index, 0);
                      int32_t res1 = cfu_op0(4, C_index, 1);
                      int32_t res2 = cfu_op0(4, C_index, 2);
                      int32_t res3 = cfu_op0(4, C_index, 3);
                      
                      if (m + i < M) [[likely]]   { 
                          int row_idx = (m + i) * N;
                          // Check N boundaries
                          if (n + j < N)     result_buffer[row_idx + (n + j)]     += res0;
                          if (n + j + 1 < N) result_buffer[row_idx + (n + j + 1)] += res1;
                          if (n + j + 2 < N) result_buffer[row_idx + (n + j + 2)] += res2;
                          if (n + j + 3 < N) result_buffer[row_idx + (n + j + 3)] += res3;
                      }
                      C_index++; 
                  }
              }

            //   printf("C_index = %d\n", C_index);
          } // End k
      } // End n
  } // End m
//   printf("cnt = %d\n", cnt);

  // =================================================================
  // STEP 4: Post-processing (保持不變)
  // =================================================================
    

    // 2. 預計算 (Pre-calculation)
    // 將 "Bias + Filter Offset" 的乘法移出主迴圈
    for (int m = 0; m < output_depth; ++m) {
        int32_t bias_val = (bias_data) ? bias_data[m] : 0;
        effective_bias[m] = bias_val + filter_sum_per_row[m] * input_offset;
    }

  for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int m = 0; m < output_depth; ++m) {
              int n = out_y * output_width + out_x;
              int32_t acc = result_buffer[m * N + n];
            //   acc += filter_sum_per_row[m] * input_offset;
            //   if (bias_data) acc += bias_data[m];
              acc += effective_bias[m];
            //   acc = my_MultiplyByQuantizedMultiplier(acc, output_multiplier[m], output_shift[m]);
            //   acc += output_offset;
              cfu_op0(8, acc, output_offset);
              acc = cfu_op0(9, output_multiplier[m], output_shift[m]);            
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[my_Offset(output_shape, 0, out_y, out_x, m)] = static_cast<int8_t>(acc);
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
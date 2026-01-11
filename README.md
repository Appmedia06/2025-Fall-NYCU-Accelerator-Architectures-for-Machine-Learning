# 2025 NYCU AAML

## Lab1: Environment Setup and Profiling a Model
* 環境安裝
## Lab2: Quantization and SIMD MAC
* 在 `conv.h` 與 `fully_connected.h` 實現 SIMD
## Lab 3: Systolic Array
* 實現 output stationary 的 systolic array
* 支援矩陣乘法(M*K x KxN)，M, K, N ∈ [4, 128)
## Lab 4: Elementwise Unit
* 加速 Logistic 和 Softmax function
* 使用 lookup table
## Lab 5: Systolic Array with im2col for Convolution
* 將 lab3 的 systolic array 移植到 `conv.h`
* 利用 im2col，將 2D convolution 變成兩個矩陣相乘

## Final Project: Pruned Wav2Letter
* 加速 Wav2Letter model (automatic speech recognition)
* 將 lab5 的 `conv.h` 移植過來，並將 MNK 的上限設置成 252 * 624 * 148
* 一些平行化處理與軟體優化
* Baseline:
  * Latency: 450000ms, Accuracy: 72.22%​
* Our method
  * Latency: 17146ms, Accuracy: 88.89%

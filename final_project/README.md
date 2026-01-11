# Final Project: Pruned Wav2Letter

## Update Log

- 2025/11/25
  - 加入 TPU
  - latyency = 86673 ms
  - accuracy = 75 %
 
- 2025/11/30
  - 新增 `my_MultiplyByQuantizedMultiplier` 用 cfu 加速
    - `SaturatingRoundingDoublingHighMul`
    - `RoundingDivideByPOT`
  - 固定常數 in `conv.h` 與 `leaky_relu.h`
  - Unroll pre-pack loop
  - Cnvlutin, 無效
  - 發現 A, B 一起傳在 lab5 會比較快 (perf cycle count 低)，但在 final project 的 latency 會比較高 - why ?
  - latyency = 79635 ms
  - accuracy = 72.22 %
 
- 2025/12/11
  - 把 tile size 從 16x16 變成 128x128，perf 大躍進
  - 目前 MNK 的 bitwidth 是 8 bit，也就是說不能 256
  - 但是我試 240 有問題，可能是 TPU 內位元沒設好
  - 若要繼續大，需要改 buffer 的 address width
  - latyency = 36608 ms
  - accuracy = 72.22 %
 
- 2025/12/12
  - 發現 TPU 裡一些邊界檢查的位元設太小，導致 overflow
  - 修改後便可以把 tile size 從 128x128 變成 256x256
  - latyency = 29682 ms
  - accuracy = 72.22 %
 
- 2025/12/13
  - 更改 `MultiplyByQuantizedMultiplier` 函式的實作能把準確率從 72.22% 提高到 88.89%
  - 使 TPU 開始計算時間重疊於輸入 B buffer 的時間
  - latyency = 28394 ms
  - accuracy = 88.89 %

- TODO
  - TPU MKN 繼續改大 (done)
  - `conh.h` 軟體優化
  - sparse 優化
  - TPU 計算重疊輸入 (done)
  - multi-TPU 
  - 發現 accuaracy 可以上升到 89% 的方法，這樣就可能可以改模型!
  - 路徑 `CFU-playground/soc/build/digilent_nexys4ddr.AAML-2025-Project/gateware/digilent_nexys4ddr_utilization_place` 裡有合成的資訊，研究如何把 buffer size 開大一點

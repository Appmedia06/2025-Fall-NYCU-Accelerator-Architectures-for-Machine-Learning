// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "TPU.v"
`include "global_buffer_bramA.v"
`include "global_buffer_bramB.v"
`include "global_buffer_bramC.v"

module Cfu (
  input               cmd_valid,
  output reg          cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

//------------------------------------------------------------------------
// Local parameter
//------------------------------------------------------------------------

localparam IDLE   = 4'd0;
localparam READ_A = 4'd1;
localparam READ_B = 4'd2;
localparam CALC   = 4'd3;
localparam RET_0  = 4'd4;
localparam DONE   = 4'd5;
localparam RET_1  = 4'd6;
localparam SAT    = 4'd7;
localparam ROU    = 4'd8;
localparam READ   = 4'd9;
localparam MUL0   = 4'd10;
localparam MUL1   = 4'd11;
localparam CMP    = 4'd12;
localparam RST    = 4'd13;
localparam RET_2  = 4'd14;

localparam FUNC_READ_A = 7'd1;
localparam FUNC_READ_B = 7'd2;
localparam FUNC_CALC   = 7'd3;
localparam FUNC_RET    = 7'd4;
localparam FUNC_SAT    = 7'd5;
localparam FUNC_ROU    = 7'd6;
localparam FUNC_READ   = 7'd7;
localparam FUNC_MUL0   = 7'd8;
localparam FUNC_MUL1   = 7'd9;
localparam FUNC_CMP    = 7'd10;
localparam FUNC_RST    = 7'd11;

localparam OFF_0 = 2'd0;
localparam OFF_1 = 2'd1;
localparam OFF_2 = 2'd2;
localparam OFF_3 = 2'd3;

//------------------------------------------------------------------------
// Combinational logic
//------------------------------------------------------------------------

reg [3:0] curr_state, next_state;

reg [31:0] in_0;
reg [31:0] in_1;

wire [6:0] funct7 = cmd_payload_function_id[9:3];

wire cmd_en = cmd_valid && cmd_ready;

wire in_valid = ( B_read_cnt == 16'd5000 );
// wire in_valid = cmd_en && ( funct7 == FUNC_CALC );

wire [15:0] cfu_index = in_0[15:0];

wire        A_wr_en   = ( curr_state == READ_A || ( curr_state == IDLE && cmd_en && funct7 == FUNC_READ_A ));
wire        A_mux_sel = ( curr_state == READ_A );
wire [15:0] A_index   = A_read_cnt;
wire [31:0] A_data_in = ( A_mux_sel ) ? in_1 : cmd_payload_inputs_0;
wire [31:0] A_data_out;

wire        B_wr_en   = ( curr_state == READ_B || ( curr_state == IDLE && cmd_en && funct7 == FUNC_READ_B ));
wire [15:0] B_index   = B_read_cnt;
wire        B_mux_sel = ( curr_state == READ_B );
wire [31:0] B_data_in = ( B_mux_sel ) ? in_1 : cmd_payload_inputs_0;
wire [31:0] B_data_out;

wire         busy;

wire [15:0]  tpu_A_index;
wire [15:0]  tpu_B_index;
wire [15:0]  tpu_C_index;

wire         C_wr_en;
wire [15:0]  C_index = ( C_index_mux_sel ) ? cfu_index : tpu_C_index;
wire [127:0] tpu_C_data_in;
wire [127:0] C_data_out;


wire A_index_mux_sel = ( curr_state == READ_A ) ? 1'b1 : 1'b0;
wire B_index_mux_sel = ( curr_state == READ_B ) ? 1'b1 : 1'b0;
wire C_index_mux_sel = ( curr_state == CALC )   ? 1'b1 : 1'b0;

reg [31:0] cfu_out;

always@ (*) begin
  case ( in_1[1:0] )
    OFF_0: cfu_out = C_data_out[127:96];
    OFF_1: cfu_out = C_data_out[95:64];
    OFF_2: cfu_out = C_data_out[63:32];
    OFF_3: cfu_out = C_data_out[31:0];
    default: cfu_out = 32'd0;
  endcase
end


wire [31:0] final_out = ( curr_state == MUL1 ) ? mul_result 
                      :                          cfu_out;                      

reg [31:0] final_out_reg;
always@ ( posedge clk ) begin
  if ( reset ) begin
    final_out_reg <= 32'b0;
  end 
  else begin
    final_out_reg <= final_out;
  end
end

wire start_ret = ( !busy ) || ( tpu_C_index >= 14'd9000 );

// --------------------------------------------------------
// MultiplyByQuantizedMultiplier
// --------------------------------------------------------

reg signed [31:0] x;
wire x_en = ( cmd_en && funct7 == FUNC_MUL0 );

always@ ( posedge clk ) begin
  if ( reset ) begin
    x <= 32'b0;
  end 
  else if ( x_en ) begin
    x <= cmd_payload_inputs_0;
  end
end

reg signed [31:0] output_offset;
always@ ( posedge clk ) begin
  if ( reset ) begin
    output_offset <= 32'd0;
  end 
  else if ( curr_state == MUL0 ) begin
    output_offset <= in_1;
  end
end

wire signed [31:0] quantized_multiplier = in_0;
wire signed [31:0] shift = in_1;
wire signed [31:0] mul_result;

// --------------------------------------------------------
// Step 1: 計算總位移量 (total_shift = 31 - shift)
// --------------------------------------------------------
// 因為 shift 是 32-bit 有號數，我們用 32-bit 進行減法運算以確保正負號正確。
wire signed [31:0] total_shift_raw;
assign total_shift_raw = 32'sd31 - shift;

// 我們只需要低 6 bits 來控制 64-bit 的位移操作 (2^6 = 64)
// 根據 TFLite 規範，shift 範圍保證 total_shift 在 [1, 62] 之間，
// 所以截取低 6 位是安全的。
wire [5:0] total_shift;
assign total_shift = total_shift_raw[5:0];

// --------------------------------------------------------
// Step 2: 計算捨入項 (round = 1 << (total_shift - 1))
// --------------------------------------------------------
wire [63:0] round_val;
assign round_val = 64'd1 << (total_shift - 1'b1);

// --------------------------------------------------------
// Step 3: 乘法運算 (x * quantized_multiplier)
// --------------------------------------------------------
wire signed [63:0] product;
assign product = x * quantized_multiplier;

// --------------------------------------------------------
// Step 4: 加法運算 (product + round)
// --------------------------------------------------------
wire signed [63:0] product_rounded;
assign product_rounded = product + $signed(round_val);

// --------------------------------------------------------
// Step 5: 算術右移 (result >> total_shift)
// --------------------------------------------------------
wire signed [63:0] shifted_result;
assign shifted_result = product_rounded >>> total_shift;

// --------------------------------------------------------
// Step 6: 輸出截斷
// --------------------------------------------------------
assign mul_result = shifted_result[31:0] + output_offset;

// wire signed [31:0] acc = mul_result;
// wire signed [31:0] minus_127 = -32'sd128;

// wire signed [31:0] t0 = ( acc > minus_127 ) ? acc : minus_127;
// wire signed [31:0] cmp_result = ( t0 < 32'sd127 ) ? ( t0 ) : 32'sd127;

// --------------------------------------------------------
// Compare
// --------------------------------------------------------
wire signed [31:0] acc = in_0 + in_1;
wire signed [31:0] minus_127 = -32'd127;

wire signed [31:0] t0 = ( acc > minus_127 ) ? acc : minus_127;
wire signed [31:0] cmp_result = ( t0 < 32'd128 ) ? ( acc ) : 32'd128;



//------------------------------------------------------------------------
// Sequential logic
//------------------------------------------------------------------------

always@ ( posedge clk ) begin
  if ( reset ) begin
    in_0 <= 32'b0;
    in_1 <= 32'b0;
  end 
  else if ( cmd_en ) begin
    in_0 <= cmd_payload_inputs_0;
    in_1 <= cmd_payload_inputs_1;
  end
end


reg [15:0] A_read_cnt;
always@ ( posedge clk ) begin
  if ( reset || ( A_read_cnt == 16'd39312 ) ) begin
    A_read_cnt <= 16'd0;
  end 
  else if ( A_wr_en ) begin
    A_read_cnt <= A_read_cnt + 1'b1;
  end
end

reg [15:0] B_read_cnt;
always@ ( posedge clk ) begin
  if ( reset || ( B_read_cnt == 16'd23088 ) ) begin
    B_read_cnt <= 16'd0;
  end 
  else if ( B_wr_en ) begin
    B_read_cnt <= B_read_cnt + 1'b1;
  end
end

//------------------------------------------------------------------------
// BRAM
//------------------------------------------------------------------------

wire gb_A_wen = ( A_index < 16'd32000 && A_wr_en );
wire gb_B_wen = ( B_index < 16'd19000 && B_wr_en );

wire gb_A_ren = ( tpu_A_index < 16'd32000 );
wire gb_B_ren = ( tpu_B_index < 16'd19000 ); 
wire gb_C_ren = 1'b1;

global_buffer_bramA #(
  .ADDR_BITS(16), 
  .DATA_BITS(32)  
)
gbuff_A (
  .clk      (clk),
  .rst_n    (!reset),
  .wr_en    (gb_A_wen),
  .ren      (gb_A_ren),
  .index_r  (tpu_A_index),
  .index_w  (A_index),  
  .data_in  (A_data_in),
  .data_out (A_data_out)
);

global_buffer_bramB #(
  .ADDR_BITS(16), 
  .DATA_BITS(32)  
)
gbuff_B (
  .clk      (clk),
  .rst_n    (!reset),
  .wr_en    (gb_B_wen),
  .ren      (gb_B_ren),  
  .index_r  (tpu_B_index),
  .index_w  (B_index),  
  .data_in  (B_data_in),
  .data_out (B_data_out)
);

global_buffer_bramC #(
  .ADDR_BITS(15),
  .DATA_BITS(128)
)
gbuff_C (
  .clk      (clk),
  .rst_n    (!reset),
  .wr_en    (C_wr_en),
  .ren      (gb_C_ren),  
  .index_r  (cfu_index),
  .index_w  (tpu_C_index),  
  .data_in  (tpu_C_data_in),
  .data_out (C_data_out)
);


reg [31:0] A_buff [8000:0];
reg [31:0] B_buff [5000:0];

wire A_buff_wen = ( A_index >= 16'd32000 && A_wr_en );
wire B_buff_wen = ( B_index >= 16'd19000 && B_wr_en );

wire A_buff_ren = ( tpu_A_index >= 16'd32000 );
wire B_buff_ren = ( tpu_B_index >= 16'd19000 );

reg [31:0] A_out;
reg [31:0] B_out;

wire [15:0] A_buff_idx_w = ( A_buff_wen ) ? A_index - 16'd32000 : 0;
wire [15:0] B_buff_idx_w = ( B_buff_wen ) ? B_index - 16'd19000 : 0;

wire [15:0] A_buff_idx_r = ( A_buff_ren ) ? tpu_A_index - 16'd32000 : 0;
wire [15:0] B_buff_idx_r = ( B_buff_ren ) ? tpu_B_index - 16'd19000 : 0;

always @ ( negedge clk ) begin
  if( A_buff_wen ) begin
    A_buff[A_buff_idx_w] <= A_data_in;
  end
  if ( A_buff_ren ) begin
    A_out <= A_buff[A_buff_idx_r];
  end
end

always @ ( negedge clk ) begin
  if( B_buff_wen ) begin
    B_buff[B_buff_idx_w] <= B_data_in;
  end
  if ( B_buff_ren ) begin
    B_out <= B_buff[B_buff_idx_r];
  end
end

wire [31:0] A_mux_out = ( A_buff_ren ) ? A_out : A_data_out;
wire [31:0] B_mux_out = ( B_buff_ren ) ? B_out : B_data_out;

//------------------------------------------------------------------------
// TPU
//------------------------------------------------------------------------

TPU tpu (
  .clk        (clk),
  .rst_n      (!reset),

  .in_valid   (in_valid),
  .K          (15'd624),
  .M          (15'd252),
  .N          (15'd148),
  .busy       (busy),

  .A_wr_en    (),
  .A_index    (tpu_A_index),
  .A_data_in  (),
  .A_data_out (A_mux_out),

  .B_wr_en    (),
  .B_index    (tpu_B_index) ,
  .B_data_in  (),
  .B_data_out (B_mux_out),

  .C_wr_en    (C_wr_en),
  .C_index    (tpu_C_index),
  .C_data_in  (tpu_C_data_in),
  .C_data_out ()
);

//------------------------------------------------------------------------
// FSM
//------------------------------------------------------------------------


always@ ( posedge clk ) begin
    if ( reset ) begin
      curr_state <= IDLE;
    end
    else begin
      curr_state <= next_state;
    end
end

always@ (*) begin
    next_state = curr_state;
    case ( curr_state )
      IDLE  : begin
        if ( cmd_en ) begin
          case ( funct7 )
            FUNC_READ_A:  next_state = READ_A;
            FUNC_READ_B:  next_state = READ_B;
            FUNC_CALC:    next_state = CALC;
            FUNC_RET:     next_state = ( busy ) ? RET_0 : RET_2;
            FUNC_SAT:     next_state = SAT;
            FUNC_ROU:     next_state = ROU;
            FUNC_READ:    next_state = READ;
            FUNC_MUL0:    next_state = MUL0;
            FUNC_MUL1:    next_state = MUL1;
            FUNC_CMP:     next_state = CMP;
            FUNC_RST:     next_state = RST;
            default: next_state = IDLE;
          endcase
        end
      end
      READ_A: next_state = IDLE;
      READ_B: next_state = IDLE;
      CALC: next_state = ( busy ) ? CALC : DONE;
      DONE: next_state = IDLE;
      RET_0: next_state = ( start_ret ) ? RET_1 : RET_0;
      // RET_0: next_state = ( busy ) ? RET_0 : RET_1;
      RET_1: next_state = IDLE;
      SAT: next_state = RET_1;
      ROU: next_state = RET_1;
      READ: next_state = IDLE;
      MUL0: next_state = IDLE;
      MUL1: next_state = RET_1;
      CMP: next_state = IDLE;
      RST: next_state = IDLE;
      RET_2: next_state = IDLE;
      default: next_state = IDLE;
    endcase
end


always@ (*) begin
    cmd_ready             = 1'b0;
    rsp_valid             = 1'b0;
    rsp_payload_outputs_0 = 32'd0;
    case ( curr_state )
      IDLE: begin
        cmd_ready = 1'b1;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end

      READ_A: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end   

      READ_B: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end        

      READ: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end              

      CALC: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end                            

      DONE: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end       

      RET_0: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end  

      RET_1: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = final_out_reg;
      end         

      SAT: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end        

      ROU: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end       

      MUL0: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end        

      MUL1: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b0;
        rsp_payload_outputs_0 = 32'd0;
      end         
      
      CMP: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = cmp_result;
      end       

      RST: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end                   

      RET_2: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = final_out;
      end

    endcase
end



endmodule
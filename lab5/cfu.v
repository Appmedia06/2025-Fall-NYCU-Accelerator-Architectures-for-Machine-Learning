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
`include "global_buffer_bram.v"

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

localparam IDLE    = 4'd0;
localparam READ_A  = 4'd1;
localparam READ_B  = 4'd2;
localparam CALC    = 4'd3;
localparam RET_0   = 4'd4;
localparam DONE    = 4'd5;
localparam RET_1   = 4'd6;
localparam SAT     = 4'd7;
localparam ROU     = 4'd8;
localparam READ    = 4'd9;
localparam MNK     = 4'd10;


localparam FUNC_READ_A = 7'd1;
localparam FUNC_READ_B = 7'd2;
localparam FUNC_CALC   = 7'd3;
localparam FUNC_RET    = 7'd4;
localparam FUNC_SAT    = 7'd5;
localparam FUNC_ROU    = 7'd6;
localparam FUNC_READ   = 7'd7;
localparam FUNC_MNK    = 7'd8;

localparam OFF_0 = 2'd0;
localparam OFF_1 = 2'd1;
localparam OFF_2 = 2'd2;
localparam OFF_3 = 2'd3;

//------------------------------------------------------------------------
// MKN
//------------------------------------------------------------------------

reg [31:0] matrix_size;

wire MNK_en = ( curr_state == MNK );


always@ ( posedge clk ) begin
  if ( reset ) begin
    matrix_size <= 32'b0;
  end 
  else if ( MNK_en ) begin
    matrix_size <= cmd_payload_inputs_0;
  end
end

wire [7:0] K = matrix_size[7:0];
wire [7:0] N = matrix_size[15:8];
wire [7:0] M = matrix_size[23:16];

//------------------------------------------------------------------------
// Combinational logic
//------------------------------------------------------------------------

reg [3:0] curr_state, next_state;

reg [31:0] in_0;
reg [31:0] in_1;

wire [6:0] funct7 = cmd_payload_function_id[9:3];

wire cmd_en = cmd_valid && cmd_ready;

wire in_valid = cmd_en && ( funct7 == FUNC_CALC );

wire [11:0] cfu_index = in_0[11:0];

wire        A_wr_en   = ( curr_state == READ_A );
wire [11:0] A_index   = ( A_index_mux_sel ) ? cfu_index : tpu_A_index;
wire [31:0] A_data_in = in_1;
wire [31:0] A_data_out;

wire        B_wr_en   = ( curr_state == READ_B );
wire [11:0] B_index   = ( B_index_mux_sel ) ? cfu_index : tpu_B_index;
wire [31:0] B_data_in = in_1;
wire [31:0] B_data_out;

wire         busy;

wire [11:0]  tpu_A_index;
wire [11:0]  tpu_B_index;
wire [11:0]  tpu_C_index;

wire         C_wr_en;
wire [11:0]  C_index = ( C_index_mux_sel ) ? cfu_index : tpu_C_index;
wire [127:0] tpu_C_data_in;
wire [127:0] C_data_out;


wire A_index_mux_sel = ( curr_state == READ_A ) ? 1'b1 : 1'b0;
wire B_index_mux_sel = ( curr_state == READ_B ) ? 1'b1 : 1'b0;
wire C_index_mux_sel = ( curr_state != CALC )   ? 1'b1 : 1'b0;

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

wire [1:0] out_sel = ( curr_state == SAT ) ? 2'd0 
                   : ( curr_state == ROU ) ? 2'd1 
                   :                         2'd2;
wire [31:0] final_out = ( out_sel == 2'd0 ) ? sat_result 
                      : ( out_sel == 2'd1 ) ? rou_result
                      : ( out_sel == 2'd2 ) ? cfu_out
                      :                       cfu_out;

reg [31:0] final_out_reg;
always@ ( posedge clk ) begin
  if ( reset ) begin
    final_out_reg <= 32'b0;
  end 
  else begin
    final_out_reg <= final_out;
  end
end

//------------------------------------------------------------------------
// SaturatingRoundingDoublingHighMul (bit-exact to C++ version)
//------------------------------------------------------------------------

wire signed [31:0] a = in_0;
wire signed [31:0] b = in_1;

// C++: overflow = (a == b) && (a == INT32_MIN)
wire overflow = (a == b) && (a == 32'sh8000_0000);

//------------------------------------------------------------------------
// Step 3: do 64-bit signed multiply (must sign-extend)
//------------------------------------------------------------------------
wire signed [63:0] a_64 = {{32{a[31]}}, a};
wire signed [63:0] b_64 = {{32{b[31]}}, b};
wire signed [63:0] ab_64 = a_64 * b_64;

//------------------------------------------------------------------------
// Step 4: compute nudge
//------------------------------------------------------------------------
wire signed [31:0] nudge_pos = 32'sd1073741824;       // 1 << 30
wire signed [31:0] nudge_neg = 32'sd1 - 32'sd1073741824;
wire signed [31:0] nudge     = (ab_64 >= 0) ? nudge_pos : nudge_neg;

//------------------------------------------------------------------------
// Step 5: C++ exact behavior: (ab_64 + nudge) / (2^31)
// Verilog >>> gives floor() for negative numbers, but C++ truncates toward 0.
// We must correct only when:
//   - added < 0
//   - added is NOT divisible by (1 << 31)
//------------------------------------------------------------------------
wire signed [63:0] added = ab_64 + {{32{nudge[31]}}, nudge};

// arithmetic shift (floor for negative numbers)
wire signed [63:0] shifted = added >>> 31;

// Detect exact divisibility: true only if low 31 bits are all zero
wire divisible = (added[30:0] == 31'd0);

// Determine if correction needed:
// C++ result = trunc toward zero
// Verilog >>> = floor for negatives → too small by 1 when not divisible
wire need_fix = (added < 0) && !divisible;

// Apply correction
wire signed [31:0] ab_x2_high32 =
    need_fix ? (shifted[31:0] + 32'sd1) : shifted[31:0];

//------------------------------------------------------------------------
// Final saturating output
//------------------------------------------------------------------------
wire signed [31:0] sat_result =
    overflow ? 32'sh7FFF_FFFF : ab_x2_high32;

//------------------------------------------------------------------------
// RoundingDivideByPOT
// x: signed 32-bit
// exponent: 0~31
//------------------------------------------------------------------------

wire signed [31:0] x = in_0;
wire [4:0] exponent = in_1[4:0];

//------------------------------------------------------------------------
// mask = (1 << exponent) - 1
//------------------------------------------------------------------------
wire [31:0] mask = (exponent == 0) ? 32'd0 :
                   ((32'h1 << exponent) - 32'd1);

//------------------------------------------------------------------------
// remainder = x & mask
//------------------------------------------------------------------------
wire [31:0] remainder = x & mask;

//------------------------------------------------------------------------
// threshold = (mask >> 1) + (x < 0 ? 1 : 0)
//------------------------------------------------------------------------
wire [31:0] half_mask = mask >> 1;
wire x_is_negative = x[31];  // MSB
wire [31:0] threshold = half_mask + (x_is_negative ? 32'd1 : 32'd0);

//------------------------------------------------------------------------
// base = x >> exponent   (arithmetic shift, matches C++)
//------------------------------------------------------------------------
wire signed [31:0] base = x >>> exponent;

//------------------------------------------------------------------------
// add_one = (remainder > threshold) ? 1 : 0
//------------------------------------------------------------------------
wire [31:0] add_one = (remainder > threshold) ? 32'd1 : 32'd0;

//------------------------------------------------------------------------
// result = base + add_one
//------------------------------------------------------------------------
wire signed [31:0] rou_result = base + add_one;

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

wire read_cnt_en = ( curr_state == READ );
reg [5:0] read_cnt;
always@ ( posedge clk ) begin
  if ( reset ) begin
    read_cnt <= 6'd0;
  end 
  else if ( read_cnt_en ) begin
    read_cnt <= read_cnt + 1'b1;
  end
end

//------------------------------------------------------------------------
// BRAM
//------------------------------------------------------------------------


global_buffer_bram #(
  .ADDR_BITS(16), 
  .DATA_BITS(32)  
)
gbuff_A (
  .clk      (clk),
  .rst_n    (!reset),
  .ram_en   (1'b1),
  .wr_en    (A_wr_en),
  .index    (A_index),
  .data_in  (A_data_in),
  .data_out (A_data_out)
);

global_buffer_bram #(
  .ADDR_BITS(16), 
  .DATA_BITS(32)  
)
gbuff_B (
  .clk      (clk),
  .rst_n    (!reset),
  .ram_en   (1'b1),
  .wr_en    (B_wr_en),
  .index    (B_index),
  .data_in  (B_data_in),
  .data_out (B_data_out)
);

global_buffer_bram #(
  .ADDR_BITS(16),
  .DATA_BITS(128)
)
gbuff_C (
  .clk      (clk),
  .rst_n    (!reset),
  .ram_en   (1'b1),
  .wr_en    (C_wr_en),
  .index    (C_index),
  .data_in  (tpu_C_data_in),
  .data_out (C_data_out)
);


//------------------------------------------------------------------------
// TPU
//------------------------------------------------------------------------

TPU tpu (
  .clk        (clk),
  .rst_n      (!reset),

  .in_valid   (in_valid),
  .K          (K),
  .M          (M),
  .N          (N),
  .busy       (busy),

  .A_wr_en    (),
  .A_index    (tpu_A_index),
  .A_data_in  (),
  .A_data_out (A_data_out),

  .B_wr_en    (),
  .B_index    (tpu_B_index) ,
  .B_data_in  (),
  .B_data_out (B_data_out),

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
            FUNC_RET:     next_state = RET_0;
            FUNC_SAT:     next_state = SAT;
            FUNC_ROU:     next_state = ROU;
            FUNC_READ:    next_state = READ;
            FUNC_MNK:     next_state = MNK;
            default: next_state = IDLE;
          endcase
        end
      end
      READ_A: next_state = IDLE;
      READ_B: next_state = IDLE;
      CALC: next_state = ( busy ) ? CALC : DONE;
      DONE: next_state = IDLE;
      RET_0: next_state = RET_1;
      RET_1: next_state = IDLE;
      SAT: next_state = RET_1;
      ROU: next_state = RET_1;
      READ: next_state = IDLE;
      MNK: next_state = IDLE;

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

      MNK: begin
        cmd_ready = 1'b0;
        rsp_valid = 1'b1;
        rsp_payload_outputs_0 = 32'd0;
      end      


    endcase
end



endmodule

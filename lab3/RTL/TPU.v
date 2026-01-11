`include "PE.v"

module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);


input            clk;
input            rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [15:0]    C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;



//----------------------------------------------------------------------
// Matric parameter
//----------------------------------------------------------------------  

reg [7:0] K_reg;
reg [7:0] M_reg;
reg [7:0] N_reg;

always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        K_reg <= 8'd0;
        M_reg <= 8'd0;
        N_reg <= 8'd0;
    end 
    else if ( param_en ) begin
        K_reg <= K;
        M_reg <= M;
        N_reg <= N;
    end
end


//----------------------------------------------------------------------
// Matric A
//----------------------------------------------------------------------  

wire [1:0] M_remainer = M_reg % 4;
wire [1:0] N_remainer = N_reg % 4;

reg [31:0] A_buffer_0;
reg [31:0] A_buffer_1;
reg [31:0] A_buffer_2;
reg [31:0] A_buffer_3;


reg [7:0] a0, a1, a2, a3;
// wire M_border = ( M_cnt_prev == M_blocks - 1 ) && M_blocks != 8'd1;
wire M_border = ( M_cnt == M_blocks - 1 ) && M_blocks != 8'd1;
always@ (*) begin
    if ( M_border ) begin
        case ( M_remainer )
            2'b00: begin
                a0 = A_data_out[31:24];
                a1 = A_data_out[23:16];
                a2 = A_data_out[15:8];
                a3 = A_data_out[7:0];
            end           
            2'b01: begin
                a0 = A_data_out[31:24];
                a1 = 8'd0;
                a2 = 8'd0;
                a3 = 8'd0;
            end
            2'b10: begin
                a0 = A_data_out[31:24];
                a1 = A_data_out[23:16];
                a2 = 8'd0;
                a3 = 8'd0;
            end
            2'b11: begin
                a0 = A_data_out[31:24];
                a1 = A_data_out[23:16];
                a2 = A_data_out[15:8];
                a3 = 8'd0;
            end            
        endcase
    end
    else begin
        a0 = A_data_out[31:24];
        a1 = A_data_out[23:16];
        a2 = A_data_out[15:8];
        a3 = A_data_out[7:0];
    end
end

always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        A_buffer_0 <= 32'b0;
        A_buffer_1 <= 32'b0;
        A_buffer_2 <= 32'b0;
        A_buffer_3 <= 32'b0;
    end 
    else if ( A_buffer_clr ) begin
        A_buffer_0 <= 32'b0;
        A_buffer_1 <= 32'b0;
        A_buffer_2 <= 32'b0;
        A_buffer_3 <= 32'b0;
    end
    else if ( A_buffer_full ) begin
        A_buffer_0 <= ( A_buffer_0 >> 8 );
        A_buffer_1 <= ( A_buffer_1 >> 8 );
        A_buffer_2 <= ( A_buffer_2 >> 8 );
        A_buffer_3 <= ( A_buffer_3 >> 8 );
    end
    else if ( A_buffer_en ) begin
        A_buffer_0 <= ( A_buffer_0 >> 8 ) | a0;
        // A_buffer_1 <= ( A_buffer_1 >> 8 ) | ( a1 << 8 );
        // A_buffer_2 <= ( A_buffer_2 >> 8 ) | ( a2 << 16 );
        // A_buffer_3 <= ( A_buffer_3 >> 8 ) | ( a3 << 24 );
        A_buffer_1 <= ( A_buffer_1 >> 8 ) | ( {24'd0, a1} << 8 ); 
        A_buffer_2 <= ( A_buffer_2 >> 8 ) | ( {24'd0, a2} << 16 );
        A_buffer_3 <= ( A_buffer_3 >> 8 ) | ( {24'd0, a3} << 24 );
    end
end

wire A_buffer_full = ( A_input_cnt == K_reg_A );
wire B_buffer_full = ( B_input_cnt == K_reg_B );

wire [7:0] PE11_west = A_buffer_0[7:0];
wire [7:0] PE21_west = A_buffer_1[7:0];
wire [7:0] PE31_west = A_buffer_2[7:0];
wire [7:0] PE41_west = A_buffer_3[7:0];

//----------------------------------------------------------------------
// Matric B
//----------------------------------------------------------------------

reg [31:0] B_buffer_0;
reg [31:0] B_buffer_1;
reg [31:0] B_buffer_2;
reg [31:0] B_buffer_3;


wire N_border = ( N_cnt_prev == N_blocks - 1 ) && N_blocks != 8'd1;
reg [7:0] b0, b1, b2, b3;
always@ (*) begin
    if ( N_border ) begin
        case ( N_remainer )
            2'b00: begin
                b0 = B_data_out[31:24];
                b1 = B_data_out[23:16];
                b2 = B_data_out[15:8];
                b3 = B_data_out[7:0];
            end           
            2'b01: begin
                b0 = B_data_out[31:24];
                b1 = 8'd0;
                b2 = 8'd0;
                b3 = 8'd0;
            end
            2'b10: begin
                b0 = B_data_out[31:24];
                b1 = B_data_out[23:16];
                b2 = 8'd0;
                b3 = 8'd0;
            end
            2'b11: begin
                b0 = B_data_out[31:24];
                b1 = B_data_out[23:16];
                b2 = B_data_out[15:8];
                b3 = 8'd0;
            end            
        endcase
    end
    else begin
        b0 = B_data_out[31:24];
        b1 = B_data_out[23:16];
        b2 = B_data_out[15:8];
        b3 = B_data_out[7:0];
    end
end


always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        B_buffer_0 <= 32'b0;
        B_buffer_1 <= 32'b0;
        B_buffer_2 <= 32'b0;
        B_buffer_3 <= 32'b0;
    end 
    else if ( B_buffer_clr ) begin
        B_buffer_0 <= 32'b0;
        B_buffer_1 <= 32'b0;
        B_buffer_2 <= 32'b0;
        B_buffer_3 <= 32'b0;
    end
    else if ( B_buffer_full ) begin
        B_buffer_0 <= ( B_buffer_0 >> 8 );
        B_buffer_1 <= ( B_buffer_1 >> 8 );
        B_buffer_2 <= ( B_buffer_2 >> 8 );
        B_buffer_3 <= ( B_buffer_3 >> 8 );
    end    
    else if ( B_buffer_en ) begin
        B_buffer_0 <= ( B_buffer_0 >> 8 ) | b0;
        B_buffer_1 <= ( B_buffer_1 >> 8 ) | ( b1 << 8 );
        B_buffer_2 <= ( B_buffer_2 >> 8 ) | ( b2 << 16 );
        B_buffer_3 <= ( B_buffer_3 >> 8 ) | ( b3 << 24 );
    end
end

wire [7:0] PE11_north = B_buffer_0[7:0];
wire [7:0] PE12_north = B_buffer_1[7:0];
wire [7:0] PE13_north = B_buffer_2[7:0];
wire [7:0] PE14_north = B_buffer_3[7:0];



//----------------------------------------------------------------------
// South output wire
//----------------------------------------------------------------------

wire [7:0] PE11_to_PE21;
wire [7:0] PE12_to_PE22;
wire [7:0] PE13_to_PE23;
wire [7:0] PE14_to_PE24;

wire [7:0] PE21_to_PE31;
wire [7:0] PE22_to_PE32;
wire [7:0] PE23_to_PE33;
wire [7:0] PE24_to_PE34;

wire [7:0] PE31_to_PE41;
wire [7:0] PE32_to_PE42;
wire [7:0] PE33_to_PE43;
wire [7:0] PE34_to_PE44;


//----------------------------------------------------------------------
// East output wire
//----------------------------------------------------------------------

wire [7:0] PE11_to_PE12;
wire [7:0] PE21_to_PE22;
wire [7:0] PE31_to_PE32;
wire [7:0] PE41_to_PE42;

wire [7:0] PE12_to_PE13;
wire [7:0] PE22_to_PE23;
wire [7:0] PE32_to_PE33;
wire [7:0] PE42_to_PE43;

wire [7:0] PE13_to_PE14;
wire [7:0] PE23_to_PE24;
wire [7:0] PE33_to_PE34;
wire [7:0] PE43_to_PE44;

//----------------------------------------------------------------------
// PE result
//----------------------------------------------------------------------

wire [31:0] result_11;
wire [31:0] result_12;
wire [31:0] result_13;
wire [31:0] result_14;
wire [31:0] result_21;
wire [31:0] result_22;
wire [31:0] result_23;
wire [31:0] result_24;
wire [31:0] result_31;
wire [31:0] result_32;
wire [31:0] result_33;
wire [31:0] result_34;
wire [31:0] result_41;
wire [31:0] result_42;
wire [31:0] result_43;
wire [31:0] result_44;

//----------------------------------------------------------------------
// PE instance
//----------------------------------------------------------------------

PE PE11 (clk, rst_n, flush, PE11_north,   PE11_west,    PE11_to_PE21, PE11_to_PE12, result_11);
PE PE12 (clk, rst_n, flush, PE12_north,   PE11_to_PE12, PE12_to_PE22, PE12_to_PE13, result_12);
PE PE13 (clk, rst_n, flush, PE13_north,   PE12_to_PE13, PE13_to_PE23, PE13_to_PE14, result_13);
PE PE14 (clk, rst_n, flush, PE14_north,   PE13_to_PE14, PE14_to_PE24,             , result_14);

PE PE21 (clk, rst_n, flush, PE11_to_PE21, PE21_west,    PE21_to_PE31, PE21_to_PE22, result_21);
PE PE22 (clk, rst_n, flush, PE12_to_PE22, PE21_to_PE22, PE22_to_PE32, PE22_to_PE23, result_22);
PE PE23 (clk, rst_n, flush, PE13_to_PE23, PE22_to_PE23, PE23_to_PE33, PE23_to_PE24, result_23);
PE PE24 (clk, rst_n, flush, PE14_to_PE24, PE23_to_PE24, PE24_to_PE34,             , result_24);

PE PE31 (clk, rst_n, flush, PE21_to_PE31, PE31_west,    PE31_to_PE41, PE31_to_PE32, result_31);
PE PE32 (clk, rst_n, flush, PE22_to_PE32, PE31_to_PE32, PE32_to_PE42, PE32_to_PE33, result_32);
PE PE33 (clk, rst_n, flush, PE23_to_PE33, PE32_to_PE33, PE33_to_PE43, PE33_to_PE34, result_33);
PE PE34 (clk, rst_n, flush, PE24_to_PE34, PE33_to_PE34, PE34_to_PE44,             , result_34);

PE PE41 (clk, rst_n, flush, PE31_to_PE41, PE41_west,                , PE41_to_PE42, result_41);
PE PE42 (clk, rst_n, flush, PE32_to_PE42, PE41_to_PE42,             , PE42_to_PE43, result_42);
PE PE43 (clk, rst_n, flush, PE33_to_PE43, PE42_to_PE43,             , PE43_to_PE44, result_43);
PE PE44 (clk, rst_n, flush, PE34_to_PE44, PE43_to_PE44,             ,             , result_44);



wire [127:0] result_cat1 = {result_11, result_12, result_13, result_14};
wire [127:0] result_cat2 = {result_21, result_22, result_23, result_24};
wire [127:0] result_cat3 = {result_31, result_32, result_33, result_34};
wire [127:0] result_cat4 = {result_41, result_42, result_43, result_44};



reg [127:0] result_out;
always@ (*) begin

    if ( N_reg == 8'd2 ) begin
        result_out = ( done_cnt == 2'd0 ) ? { result_cat1[127:64], 64'b0 }
                   : ( done_cnt == 2'd1 ) ? { result_cat2[127:64], 64'b0 }
                   : ( done_cnt == 2'd2 ) ? { result_cat3[127:64], 64'b0 }
                   : ( done_cnt == 2'd3 ) ? { result_cat4[127:64], 64'b0 }
                   :                        128'd0;
    end
    else if ( N_reg >= 8'd4 ) begin
        result_out = ( done_cnt == 2'd0 ) ? result_cat1
                   : ( done_cnt == 2'd1 ) ? result_cat2
                   : ( done_cnt == 2'd2 ) ? result_cat3
                   : ( done_cnt == 2'd3 ) ? result_cat4
                   :                        128'd0;
    end

end                                                   



//----------------------------------------------------------------------
// counter
//----------------------------------------------------------------------  

reg [15:0] calc_cnt;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        calc_cnt <= 16'd0;
    end 
    else if ( calc_cnt_clr ) begin
        calc_cnt <= 16'd0;
    end
    else if ( calc_cnt_en ) begin
        calc_cnt <= calc_cnt + 1'b1;
    end
end

reg [15:0] A_input_cnt;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        A_input_cnt <= 16'd0;
    end 
    else if ( A_input_cnt_set ) begin
        A_input_cnt <= (M_cnt * K_reg);
    end
    else if ( A_input_cnt_en ) begin
        A_input_cnt <= A_input_cnt + 1'b1;
    end
end

reg [15:0] B_input_cnt;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        B_input_cnt <= 16'd0;
    end 
    else if ( B_input_cnt_set ) begin
        B_input_cnt <= (N_cnt * K_reg);
    end
    else if ( B_input_cnt_en ) begin
        B_input_cnt <= B_input_cnt + 1'b1;
    end
end

wire [15:0] K_reg_A = K_reg * (M_cnt + 1);
wire [15:0] K_reg_B = K_reg * (N_cnt + 1);
wire A_input_cnt_en  = ( curr_state == CALC ) && ( A_input_cnt != K_reg_A );
wire A_input_cnt_set = ( done_over ) || ( curr_state == IDLE );

wire B_input_cnt_en  = ( curr_state == CALC ) && ( B_input_cnt != K_reg_B );
wire B_input_cnt_set = ( done_over ) || ( curr_state == IDLE );


wire [7:0] M_blocks = ((M_reg + 8'd3) >> 2);
wire [7:0] N_blocks = ((N_reg + 8'd3) >> 2);
wire [15:0] MN_block = M_blocks * N_blocks;

reg [7:0] M_cnt, N_cnt;
reg       M_or_N;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        M_cnt <= 8'd0;
        N_cnt <= 8'd0;
        M_or_N <= 1'b0;
    end 
    else if ( MN_cnt_clr ) begin
        M_cnt <= 8'd0;
        N_cnt <= 8'd0;
        M_or_N <= 1'b0;
    end
    else if ( MN_cnt_en ) begin
        if ( M_cnt >= M_blocks - 1 ) begin
            N_cnt <= N_cnt + 8'd1;
            M_cnt <= 8'd0;
            M_or_N <= 1'b0;
        end
        else begin
            M_cnt <= M_cnt + 8'd1;
            M_or_N <= 1'b1;
        end
    end
end

wire MN_cnt_en  = ( calc_over );
wire MN_cnt_clr = ( curr_state == IDLE );

reg [7:0] M_cnt_prev, N_cnt_prev;
reg       M_or_N_prev;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        M_cnt_prev <= 8'd0;
        N_cnt_prev <= 8'd0;
        M_or_N_prev <= 1'b0;
    end 
    else if ( MN_cnt_prev_clr ) begin
        M_cnt_prev <= 8'd0;
        N_cnt_prev <= 8'd0;
        M_or_N_prev <= 1'b0;
    end
    else if ( MN_cnt_prev_en ) begin
        if ( M_cnt_prev >= M_blocks - 1 ) begin
            N_cnt_prev <= N_cnt_prev + 8'd1;
            M_cnt_prev <= 8'd0;
            M_or_N_prev <= 1'b0;
        end
        else begin
            M_cnt_prev <= M_cnt_prev + 8'd1;
            M_or_N_prev <= 1'b1;
        end
    end
end

wire MN_cnt_prev_en  = ( done_over );
wire MN_cnt_prev_clr = ( curr_state == IDLE );



reg [1:0] done_cnt;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        done_cnt <= 8'd0;
    end 
    else if ( done_cnt_clr ) begin
        done_cnt <= 8'd0;
    end
    else if ( done_cnt_en ) begin
        done_cnt <= done_cnt + 1'b1;
    end
end

wire done_cnt_en = ( curr_state == DONE );
wire done_cnt_clr = ( curr_state != DONE );

reg [15:0] index_cnt;
always @( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        index_cnt <= 8'd0;
    end 
    else if ( index_cnt_clr ) begin
        index_cnt <= 8'd0;
    end
    else if ( index_cnt_en ) begin
        index_cnt <= index_cnt + 1'b1;
    end
end


wire index_cnt_en = ( curr_state == DONE ) && ( (index_cnt >> 2) < MN_block );
wire index_cnt_clr = ( curr_state == IDLE );


//----------------------------------------------------------------------
// FSM
//----------------------------------------------------------------------  

localparam IDLE      = 3'b000;
localparam CALC      = 3'b001;
localparam DONE      = 3'b010;
// localparam FLUSH     = 3'b011;
// localparam SET_INPUT = 3'b100;


reg [2:0] curr_state, next_state;

always @(posedge clk) begin
    if ( !rst_n )
        curr_state <= IDLE;
    else
        curr_state <= next_state;
end

wire [8:0] PE_limit = K_reg + 3;

wire jump_out_done_M = ( M_border && ( M_remainer != 2'd0 ) && (done_cnt == M_remainer - 1));
wire jump_out_done_N = 1'b0;
wire jump_out_done = ( M_or_N_prev ) ? jump_out_done_M : jump_out_done_N;

wire done_over = ( done_cnt == 3 || jump_out_done ) && ( curr_state == DONE );
wire calc_over = ( calc_cnt == PE_limit );

wire [1:0] M_sub = ( M_remainer  == 2'd0 ) ? 2'd0 : ( 4 - M_remainer );
wire [1:0] N_sub = ( N_remainer  == 2'd0 ) ? 2'd0 : ( 4 - N_remainer );
wire [15:0] output_times =  ( MN_block * 4 - ( N_blocks * M_sub ) );


wire output_over = (index_cnt >= output_times - 1);

always @(*) begin
    case ( curr_state )

        IDLE: next_state = ( in_valid ) ? CALC : IDLE;

        CALC: next_state = ( calc_over ) ? DONE : CALC;

        DONE: next_state = ( done_over ) ? ( output_over ? IDLE : CALC) : DONE;

        default: next_state = IDLE;

    endcase
end

reg         A_wr_en_reg;
reg [15:0]  A_index_reg;
reg [31:0]  A_data_in_reg;

reg         B_wr_en_reg;
reg [15:0]  B_index_reg;
reg [31:0]  B_data_in_reg;

reg         C_wr_en_reg;
reg [15:0]  C_index_reg;
reg [127:0] C_data_in_reg;

always @(*) begin

    busy           = 1'b0;

    A_wr_en_reg    = 1'b0;
    A_index_reg    = 16'd0;
    A_data_in_reg  = 128'd0;

    B_wr_en_reg    = 1'b0;
    B_index_reg    = 16'd0;
    B_data_in_reg  = 128'd0;


    C_wr_en_reg    = 1'b0;
    C_index_reg    = 16'd0;
    C_data_in_reg  = 128'd0;

    case (curr_state)
        IDLE: begin
            busy           = 1'b0;

            A_index_reg    = 16'd0;
            B_index_reg    = 16'd0;

            C_wr_en_reg    = 1'b0;
            C_index_reg    = 16'd0;
            C_data_in_reg = 128'd0;
        end

        CALC: begin
            busy           = 1'b1;

            A_index_reg    = A_input_cnt;
            B_index_reg    = B_input_cnt;

            C_wr_en_reg    = 1'b0;
            C_index_reg    = 16'd0;
            C_data_in_reg  = 128'd0;
        end

        DONE: begin
            busy           = 1'b1;

            A_index_reg    = A_input_cnt;
            B_index_reg    = B_input_cnt;

            C_wr_en_reg    = 1'b1;
            C_index_reg    = index_cnt;
            C_data_in_reg  = result_out;
        end
    endcase
end  

assign A_wr_en   = A_wr_en_reg;
assign A_index   = A_index_reg;
assign A_data_in = A_data_in_reg;

assign B_wr_en   = B_wr_en_reg;
assign B_index   = B_index_reg;
assign B_data_in = B_data_in_reg;

assign C_wr_en   = C_wr_en_reg;
assign C_index   = C_index_reg;
assign C_data_in = C_data_in_reg;

wire A_buffer_en = ( curr_state == CALC );
wire B_buffer_en = ( curr_state == CALC );
wire calc_cnt_en = ( curr_state == CALC );
wire calc_cnt_clr = ( curr_state != CALC );
wire param_en = ( curr_state == IDLE ) && in_valid;

wire A_buffer_clr = ( curr_state == DONE );
wire B_buffer_clr = ( curr_state == DONE );

wire flush = ( done_over );


endmodule
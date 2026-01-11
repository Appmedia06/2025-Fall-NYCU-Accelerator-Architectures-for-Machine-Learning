module global_buffer_bramB #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      wr_en,
  input                      ren,  
  input      [ADDR_BITS-1:0] index_r,
  input      [ADDR_BITS-1:0] index_w,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  // parameter DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] gbuff [19000:0];

  always @ ( negedge clk ) begin
    if(wr_en) begin
      gbuff[index_w] <= data_in;
    end
    if ( ren ) begin
      data_out <= gbuff[index_r];
    end
  end

endmodule
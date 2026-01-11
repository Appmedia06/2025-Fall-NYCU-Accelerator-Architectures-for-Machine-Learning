module PE (
    clk, 
    rst_n,
    flush,     
    in_north, 
    in_west, 
    out_south, 
    out_east, 
    result
);

    input              clk, rst_n;
    input              flush;
	input       signed [7:0]  in_north, in_west;
	output reg  signed [7:0]  out_south, out_east;
	
	output reg signed [31:0] result;
	wire       signed [31:0] multi;

	always @( posedge clk or negedge rst_n ) begin
		if( !rst_n ) begin
			result    <= 0;
			out_east  <= 0;
			out_south <= 0;
		end
        else if ( flush ) begin
			result    <= 0;
			out_east  <= 0;
			out_south <= 0;
        end
		else begin
			result    <= $signed(result) + $signed({{16{multi[15]}}, multi});
			out_east  <= in_west;
			out_south <= in_north;
		end
	end
	assign multi = $signed(in_north) * $signed(in_west);

endmodule
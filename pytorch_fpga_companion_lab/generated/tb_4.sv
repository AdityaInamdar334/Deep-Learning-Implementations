`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1, start=0;
  reg signed [7:0] a [0:7];
  reg signed [7:0] b [0:7];
  wire signed [31:0] result;
  wire done;
  dot8_lanes_4 dut(.clk(clk),.rst(rst),.start(start),.a(a),.b(b),.result(result),.done(done));
  always #5 clk = ~clk;
  initial begin
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6; a[6]=7; a[7]=8;
    b[0]=8; b[1]=7; b[2]=6; b[3]=5; b[4]=4; b[5]=3; b[6]=2; b[7]=1;
    @(negedge clk); rst=0; start=1;
    @(negedge clk); start=0;
    wait(done);
    #1;
    if(result !== 120) $fatal(1, "wrong result: %d", result);
    $display("PASS lanes=4 result=%d", result);
    $finish;
  end
  initial begin #2000; $fatal(1,"timeout"); end
endmodule

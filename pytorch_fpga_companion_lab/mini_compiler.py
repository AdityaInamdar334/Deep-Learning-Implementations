"""Educational dot-product schedule and Verilog generator. No vendor synthesis claims."""
from __future__ import annotations
import argparse
from pathlib import Path

N = 8
WIDTH = 8

def dot_ref(a, b):
    if len(a) != N or len(b) != N:
        raise ValueError('Expected eight values per vector')
    return sum(int(x) * int(y) for x, y in zip(a, b))

def scheduled_dot(a, b, lanes=1):
    if N % lanes or lanes not in (1, 2, 4, 8):
        raise ValueError('lanes must be 1, 2, 4, or 8')
    acc = 0
    trace=[]
    for start in range(0, N, lanes):
        partial = sum(int(a[k])*int(b[k]) for k in range(start, start+lanes))
        acc+=partial
        trace.append({'cycle':start//lanes, 'indices':list(range(start,start+lanes)), 'partial':partial, 'running':acc})
    return acc,trace

def render_verilog(lanes: int) -> str:
    if lanes not in (1,2,4,8):
        raise ValueError('unsupported lane count')
    groups=N//lanes
    # combinational parallel multipliers within each group; sequential group accumulation
    exprs=[]
    for g in range(groups):
        terms=[f"($signed(a[{g*lanes+i}]) * $signed(b[{g*lanes+i}]))" for i in range(lanes)]
        exprs.append(' + '.join(terms))
    cases='\n'.join(f"        {g}: partial = {expr};" for g,expr in enumerate(exprs))
    return f'''// Generated educational RTL: 8x int8 dot product, {lanes} parallel multiplier lanes.
// Behavioral functional demonstration; actual DSP/LUT mapping is synthesis-dependent.
module dot8_lanes_{lanes} (
    input wire clk, input wire rst, input wire start,
    input wire signed [7:0] a [0:7],
    input wire signed [7:0] b [0:7],
    output reg signed [31:0] result,
    output reg done
);
  reg [3:0] group_index;
  reg active;
  reg signed [31:0] partial;
  always @* begin
    partial = 0;
    case (group_index)
{cases}
      default: partial = 0;
    endcase
  end
  always @(posedge clk) begin
    if (rst) begin
      result <= 0;
      group_index <= 0;
      active <= 0;
      done <= 0;
    end else begin
      done <= 0;
      if (start && !active) begin
        result <= 0;
        group_index <= 0;
        active <= 1;
      end else if (active) begin
        result <= result + partial;
        if (group_index == {groups-1}) begin
          active <= 0;
          done <= 1;
        end else begin
          group_index <= group_index + 1;
        end
      end
    end
  end
endmodule
'''

def render_testbench(lanes:int)->str:
    return f'''`timescale 1ns/1ps
module tb;
  reg clk=0, rst=1, start=0;
  reg signed [7:0] a [0:7];
  reg signed [7:0] b [0:7];
  wire signed [31:0] result;
  wire done;
  dot8_lanes_{lanes} dut(.clk(clk),.rst(rst),.start(start),.a(a),.b(b),.result(result),.done(done));
  always #5 clk = ~clk;
  initial begin
    a[0]=1; a[1]=2; a[2]=3; a[3]=4; a[4]=5; a[5]=6; a[6]=7; a[7]=8;
    b[0]=8; b[1]=7; b[2]=6; b[3]=5; b[4]=4; b[5]=3; b[6]=2; b[7]=1;
    @(negedge clk); rst=0; start=1;
    @(negedge clk); start=0;
    wait(done);
    #1;
    if(result !== 120) $fatal(1, "wrong result: %d", result);
    $display("PASS lanes={lanes} result=%d", result);
    $finish;
  end
  initial begin #2000; $fatal(1,"timeout"); end
endmodule
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lanes',type=int,choices=(1,2,4,8),default=2)
    p.add_argument('--out',default='generated')
    args=p.parse_args()
    a=list(range(1,9)); b=list(range(8,0,-1))
    result,trace=scheduled_dot(a,b,args.lanes)
    print('reference',dot_ref(a,b),'scheduled',result,'lanes',args.lanes)
    for row in trace:print(row)
    path=Path(args.out);path.mkdir(parents=True,exist_ok=True)
    (path/f'dot8_lanes_{args.lanes}.sv').write_text(render_verilog(args.lanes))
    (path/f'tb_{args.lanes}.sv').write_text(render_testbench(args.lanes))
    print('Generated RTL and testbench in',path)
if __name__=='__main__':main()

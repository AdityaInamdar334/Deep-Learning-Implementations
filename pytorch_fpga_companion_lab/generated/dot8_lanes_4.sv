// Generated educational RTL: 8x int8 dot product, 4 parallel multiplier lanes.
// Behavioral functional demonstration; actual DSP/LUT mapping is synthesis-dependent.
module dot8_lanes_4 (
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
        0: partial = ($signed(a[0]) * $signed(b[0])) + ($signed(a[1]) * $signed(b[1])) + ($signed(a[2]) * $signed(b[2])) + ($signed(a[3]) * $signed(b[3]));
        1: partial = ($signed(a[4]) * $signed(b[4])) + ($signed(a[5]) * $signed(b[5])) + ($signed(a[6]) * $signed(b[6])) + ($signed(a[7]) * $signed(b[7]));
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
        if (group_index == 1) begin
          active <= 0;
          done <= 1;
        end else begin
          group_index <= group_index + 1;
        end
      end
    end
  end
endmodule

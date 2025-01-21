"builtin.module"() ({
  %0:3 = "test.op"() : () -> (!riscv.reg, !riscv.reg, !riscv.reg)
  "snitch_stream.streaming_region"(%0#0, %0#1, %0#2) <{operandSegmentSizes = array<i32: 2, 1>, stride_patterns = [#snitch_stream.stride_pattern<ub = [2], strides = [8]>, #snitch_stream.stride_pattern<ub = [3, 2], strides = [8, 24]>, #snitch_stream.stride_pattern<ub = [5, 4, 3, 2], strides = [8, 40, 160, 480]>]}> ({
  ^bb0(%arg2: !snitch.readable<!riscv.freg<ft0>>, %arg3: !snitch.readable<!riscv.freg<ft1>>, %arg4: !snitch.writable<!riscv.freg<ft2>>):
    "test.op"() : () -> ()
  }) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
  "snitch_stream.streaming_region"(%0#0, %0#1) <{operandSegmentSizes = array<i32: 1, 1>, stride_patterns = [#snitch_stream.stride_pattern<ub = [2], strides = [8]>]}> ({
  ^bb0(%arg0: !snitch.readable<!riscv.freg<ft0>>, %arg1: !snitch.writable<!riscv.freg<ft1>>):
    "test.op"() : () -> ()
  }) : (!riscv.reg, !riscv.reg) -> ()
}) : () -> ()

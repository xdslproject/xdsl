// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

builtin.module {
  func.func @mixed_bounds_2d(%in : !stencil.field<?x[-4,68]xf64>) {
    "func.return"() : () -> ()
  }
}

// CHECK: stencil types can only be fully dynamic or sized.

// -----

builtin.module {
  func.func @buffered_and_stored_1d(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %outt = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<[-1,68]xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    %outt_buffered = "stencil.buffer"(%outt) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: A stencil.buffer's operand temp should only be buffered. You can use stencil.buffer's output instead!

// -----

builtin.module {
  func.func @buffer_types_mismatch_1d(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %outt = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<[-1,68]xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    %outt_buffered = "stencil.buffer"(%outt) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<?xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected input and output to have the same bounds

// -----

builtin.module {
  func.func @buffer_operand_source_1d(%temp : !stencil.temp<[0,68]xf64>) {
    %outt_buffered = "stencil.buffer"(%temp) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.buffer operand to be a result of stencil.apply or stencil.combine got block argument

// -----

builtin.module {
  func.func @buffer_operand_source_1d() {
    %temp = "test.op"() : () -> !stencil.temp<[0,68]xf64>
    %outt_buffered = "stencil.buffer"(%temp) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.buffer operand to be a result of stencil.apply or stencil.combine got test.op

// -----

builtin.module {
  func.func @apply_no_return_1d(%in : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"() : () -> ()
    }) : (!stencil.temp<?xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.apply to have at least 1 result, got 0

// -----

builtin.module {
  func.func @access_bad_temp_1d(%in : !stencil.field<[-4,68]xf64>, %bigin : !stencil.field<[-4,68]x[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %bigint = "stencil.load"(%bigin) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<?x?xf64>
    %outt = "stencil.apply"(%int, %bigint) <{operandSegmentSizes = array<i32: 2, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>, %bigintb : !stencil.temp<?x?xf64>):
      %v = "stencil.access"(%bigintb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<?x?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>, !stencil.temp<?x?xf64>) -> (!stencil.temp<?xf64>)
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Operation does not verify: Expected stencil.access operand to be of rank 1 to match its parent apply, got 2

// -----

builtin.module {
  func.func @access_bad_offset_1d(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1, 1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>)
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Expected offset's rank to be 1 to match the operand's rank, got 2

// -----

builtin.module {
  func.func @access_out_of_apply_1d(%in : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %v = "stencil.access"(%int) {"offset" = #stencil.index<[0]>} : (!stencil.temp<?xf64>) -> f64
    "func.return"() : () -> ()
  }
}

 // CHECK: 'stencil.access' expects ancestor op 'stencil.apply'

// -----

builtin.module {
  func.func @wrong_return_arity_1d(%in : !stencil.field<[-4,68]xf64>, %bigin : !stencil.field<[-4,68]x[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt1, %outt2 = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<?xf64>)
    "stencil.store"(%outt1, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

 // CHECK: stencil.return expected 2 operands to match the parent stencil.apply result types, got 1

// -----

builtin.module {
  func.func @wrong_return_types_1d(%in : !stencil.field<[-4,68]xf64>, %bigin : !stencil.field<[-4,68]x[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt1, %outt2 = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v, %v) : (f64, f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<?xf32>)
    "stencil.store"(%outt1, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

 // CHECK: stencil.return expected operand types to match the parent stencil.apply result element types. Got f64 at index 1, expected f32.


// -----

builtin.module {
  func.func @different_apply_bounds(%in : !stencil.field<[-4,68]xf64>, %bigin : !stencil.field<[-4,68]x[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt1, %outt2 = "stencil.apply"(%int) <{operandSegmentSizes = array<i32: 1, 0>}> ({
    ^bb0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<[-1]>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v, %v) : (f64, f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<[0,64]xf64>)
    "stencil.store"(%outt1, %out) {"lb" = #stencil.index<[0]>, "ub" = #stencil.index<[68]>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Expected all output types bounds to be equals.

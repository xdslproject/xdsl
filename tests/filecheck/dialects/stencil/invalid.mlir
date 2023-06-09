// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
  func.func @buffered_and_stored(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %outt = "stencil.apply"(%int) ({
    ^0(%intb : !stencil.temp<[-1,68]xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    %outt_buffered = "stencil.buffer"(%outt) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<[0,68]xf64>, !stencil.field<[0,1024]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: A stencil.buffer's operand temp should only be buffered. You can use stencil.buffer's output instead!

// -----

builtin.module {
  func.func @buffer_types_mismatch(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<[-1,68]xf64>
    %outt = "stencil.apply"(%int) ({
    ^0(%intb : !stencil.temp<[-1,68]xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<-1>} : (!stencil.temp<[-1,68]xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<[-1,68]xf64>) -> !stencil.temp<[0,68]xf64>
    %outt_buffered = "stencil.buffer"(%outt) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<?xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected operand and result type to be equal, got (!stencil.temp<[0,68]xf64>) -> !stencil.temp<?xf64>

// -----

builtin.module {
  func.func @buffer_operand_source(%temp : !stencil.temp<[0,68]xf64>) {
    %outt_buffered = "stencil.buffer"(%temp) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.buffer to buffer a stencil.apply's output, got Block(_args=(<BlockArgument[!stencil.temp<[0,68]xf64>] index: 0, uses: 1>,), num_ops=2)

// -----

builtin.module {
  func.func @apply_no_return(%in : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    "stencil.apply"(%int) ({
    ^0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
    }) : (!stencil.temp<?xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.apply to have at least 1 result, got 0

// -----

builtin.module {
  func.func @access_bad_temp(%in : !stencil.field<[-4,68]xf64>, %bigin : !stencil.field<[-4,68]x[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %bigint = "stencil.load"(%bigin) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<?x?xf64>
    %outt = "stencil.apply"(%int, %bigint) ({
    ^0(%intb : !stencil.temp<?xf64>, %bigintb : !stencil.temp<?x?xf64>):
      %v = "stencil.access"(%bigintb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?x?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>, !stencil.temp<?x?xf64>) -> (!stencil.temp<?xf64>)
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Operation does not verify: Expected stencil.access operand to be of rank 1 to match its parent apply, got 2

// -----

builtin.module {
  func.func @access_bad_offset(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt = "stencil.apply"(%int) ({
    ^0(%intb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%intb) {"offset" = #stencil.index<-1, 1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> (!stencil.temp<?xf64>)
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }
}

// CHECK: Expected offset's rank to be 1 to match the operand's rank, got 2

// -----

builtin.module {
  func.func @access_out_of_apply(%in : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %v = "stencil.access"(%int) {"offset" = #stencil.index<0>} : (!stencil.temp<?xf64>) -> f64
    "func.return"() : () -> ()
  }
}

 // CHECK: 'stencil.access' expects parent op 'stencil.apply'

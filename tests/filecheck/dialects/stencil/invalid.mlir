// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
  func.func @copy_1d(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
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
  func.func @copy_1d(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[0,1024]xf64>) {
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
  func.func @copy_1d(%temp : !stencil.temp<[0,68]xf64>) {
    %outt_buffered = "stencil.buffer"(%temp) : (!stencil.temp<[0,68]xf64>) -> !stencil.temp<[0,68]xf64>
    "func.return"() : () -> ()
  }
}

// CHECK: Expected stencil.buffer to buffer a stencil.apply's output, got Block(_args=(<BlockArgument[!stencil.temp<[0,68]xf64>] index: 0, uses: 1>,), num_ops=2)

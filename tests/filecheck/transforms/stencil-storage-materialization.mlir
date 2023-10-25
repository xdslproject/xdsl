// RUN: xdsl-opt %s -p stencil-storage-materialization | filecheck %s

// This should not change with the pass applied.

builtin.module{
  func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %outt = "stencil.apply"(%int) ({
    ^0(%inb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%inb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }

// CHECK:      func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:   %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
// CHECK-NEXT:   %outt = "stencil.apply"(%int) ({
// CHECK-NEXT:   ^0(%inb : !stencil.temp<?xf64>):
// CHECK-NEXT:     %v = "stencil.access"(%inb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
// CHECK-NEXT:     "stencil.return"(%v) : (f64) -> ()
// CHECK-NEXT:   }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
// CHECK-NEXT:   "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

 // Here we want to see a buffer added after the first apply.

  func.func @buffer_copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %midt = "stencil.apply"(%int) ({
    ^0(%inb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%inb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    %outt = "stencil.apply"(%midt) ({
    ^0(%midb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%midb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }

  //CHECK:      func.func @buffer_copy(%in_1 : !stencil.field<[-4,68]xf64>, %out_1 : !stencil.field<[-4,68]xf64>) {
  //CHECK-NEXT:   %int_1 = "stencil.load"(%in_1) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
  //CHECK-NEXT:   %midt = "stencil.apply"(%int_1) ({
  //CHECK-NEXT:   ^1(%0 : !stencil.temp<?xf64>):
  //CHECK-NEXT:     %1 = "stencil.access"(%0) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
  //CHECK-NEXT:     "stencil.return"(%1) : (f64) -> ()
  //CHECK-NEXT:   }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
  //CHECK-NEXT:   %midt_1 = "stencil.buffer"(%midt) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
  //CHECK-NEXT:   %outt_1 = "stencil.apply"(%midt_1) ({
  //CHECK-NEXT:   ^2(%midb : !stencil.temp<?xf64>):
  //CHECK-NEXT:     %v_1 = "stencil.access"(%midb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
  //CHECK-NEXT:     "stencil.return"(%v_1) : (f64) -> ()
  //CHECK-NEXT:   }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
  //CHECK-NEXT:   "stencil.store"(%outt_1, %out_1) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
  //CHECK-NEXT:   func.return
  //CHECK-NEXT: }

  // Here we don't want to see a buffer added after the apply, because the result is stored.

  func.func @stored_copy(%in : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = "stencil.load"(%in) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
    %midt = "stencil.apply"(%int) ({
    ^0(%inb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%inb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%midt, %midout) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    %outt = "stencil.apply"(%midt) ({
    ^0(%midb : !stencil.temp<?xf64>):
      %v = "stencil.access"(%midb) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
      "stencil.return"(%v) : (f64) -> ()
    }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
    "stencil.store"(%outt, %out) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
    func.return
  }

// CHECK:      func.func @stored_copy(%in_2 : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out_2 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:   %int_2 = "stencil.load"(%in_2) : (!stencil.field<[-4,68]xf64>) -> !stencil.temp<?xf64>
// CHECK-NEXT:   %midt_2 = "stencil.apply"(%int_2) ({
// CHECK-NEXT:   ^3(%inb_1 : !stencil.temp<?xf64>):
// CHECK-NEXT:     %v_2 = "stencil.access"(%inb_1) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
// CHECK-NEXT:     "stencil.return"(%v_2) : (f64) -> ()
// CHECK-NEXT:   }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
// CHECK-NEXT:   "stencil.store"(%midt_2, %midout) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
// CHECK-NEXT:   %outt_2 = "stencil.apply"(%midt_2) ({
// CHECK-NEXT:   ^4(%midb_1 : !stencil.temp<?xf64>):
// CHECK-NEXT:     %v_3 = "stencil.access"(%midb_1) {"offset" = #stencil.index<-1>} : (!stencil.temp<?xf64>) -> f64
// CHECK-NEXT:     "stencil.return"(%v_3) : (f64) -> ()
// CHECK-NEXT:   }) : (!stencil.temp<?xf64>) -> !stencil.temp<?xf64>
// CHECK-NEXT:   "stencil.store"(%outt_2, %out_2) {"lb" = #stencil.index<0>, "ub" = #stencil.index<68>} : (!stencil.temp<?xf64>, !stencil.field<[-4,68]xf64>) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
}

// CHECK: }

// RUN: xdsl-opt %s -p stencil-storage-materialization | filecheck %s

// This should not change with the pass applied.

builtin.module{
  func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %outt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %inb [-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    stencil.store %outt to %out ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %outt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %inb [-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt to %out ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

 // Here we want to see a buffer added after the first apply.

  func.func @buffer_copy(%in_1 : !stencil.field<[-4,68]xf64>, %out_1 : !stencil.field<[-4,68]xf64>) {
    %int_1 = stencil.load %in_1 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %midt = stencil.apply(%inb_1 = %int_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v_1 = stencil.access %inb_1 [-1] : !stencil.temp<?xf64>
      stencil.return %v_1 : f64
    }
    %outt_1 = stencil.apply(%midb = %midt : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v_2 = stencil.access %midb [-1] : !stencil.temp<?xf64>
      stencil.return %v_2 : f64
    }
    stencil.store %outt_1 to %out_1 ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @buffer_copy(%in_1 : !stencil.field<[-4,68]xf64>, %out_1 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int_1 = stencil.load %in_1 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %midt = stencil.apply(%0 = %int_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %1 = stencil.access %0 [-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %1 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %midt_1 = stencil.buffer %midt : !stencil.temp<?xf64>
// CHECK-NEXT:      %outt_1 = stencil.apply(%midb = %midt_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v_2 = stencil.access %midb [-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v_2 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt_1 to %out_1 ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  // Here we don't want to see a buffer added after the apply, because the result is stored.

  func.func @stored_copy(%in_2 : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out_2 : !stencil.field<[-4,68]xf64>) {
    %int_2 = stencil.load %in_2 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %midt_1 = stencil.apply(%inb_2 = %int_2 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v_3 = stencil.access %inb_2 [-1] : !stencil.temp<?xf64>
      stencil.return %v_3 : f64
    }
    stencil.store %midt_1 to %midout ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    %outt_2 = stencil.apply(%midb_1 = %midt_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v_4 = stencil.access %midb_1 [-1] : !stencil.temp<?xf64>
      stencil.return %v_4 : f64
    }
    stencil.store %outt_2 to %out_2 ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @stored_copy(%in_2 : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out_2 : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int_2 = stencil.load %in_2 : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %midt_1 = stencil.apply(%inb_2 = %int_2 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v_3 = stencil.access %inb_2 [-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v_3 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %midt_1 to %midout ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %outt_2 = stencil.apply(%midb_1 = %midt_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v_4 = stencil.access %midb_1 [-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v_4 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt_2 to %out_2 ([0] : [68]) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @combine(%0 : !stencil.field<?x?xf64>) {
    %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
    %2 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
      %3 = arith.constant 1.000000e+00 : f64
      stencil.return %3 : f64
    }
    %4 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
      %5 = arith.constant 2.000000e+00 : f64
      stencil.return %5 : f64
    }
    %6 = stencil.combine 0 at 33 lower = (%2 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%4 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
    %7 = stencil.apply(%8 = %6 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
      %9 = arith.constant 1.000000e+00 : f64
      %10 = stencil.access %8 [0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
      %11 = arith.addf %9, %10 : f64
      stencil.return %11 : f64
    }
    stencil.store %7 to %1 ([1, 2] : [65, 66]) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @combine(%2 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %3 = stencil.cast %2 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      %4 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
// CHECK-NEXT:        %5 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        stencil.return %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
// CHECK-NEXT:        %7 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        stencil.return %7 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %8 = stencil.combine 0 at 33 lower = (%4 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%6 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:      %9 = stencil.buffer %8 : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:      %10 = stencil.apply(%11 = %9 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
// CHECK-NEXT:        %12 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %13 = stencil.access %11 [0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:        %14 = arith.addf %12, %13 : f64
// CHECK-NEXT:        stencil.return %14 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %10 to %3 ([1, 2] : [65, 66]) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}

// CHECK-NEXT: }

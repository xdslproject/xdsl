// RUN: xdsl-opt %s -p stencil-storage-materialization | filecheck %s

// This should not change with the pass applied.

builtin.module{
  func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %outt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %outt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

 // Here we want to see a buffer added after the first apply.

  func.func @buffer_copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %midt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    %outt = stencil.apply(%midb = %midt : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %midb[-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @buffer_copy(%in : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %midt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %midt_1 = stencil.buffer %midt : !stencil.temp<?xf64>
// CHECK-NEXT:      %outt = stencil.apply(%midb = %midt_1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %midb[-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  // Here we don't want to see a buffer added after the apply, because the result is stored.

  func.func @stored_copy(%in : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
    %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
    %midt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    stencil.store %midt to %midout(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    %outt = stencil.apply(%midb = %midt : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
      %v = stencil.access %midb[-1] : !stencil.temp<?xf64>
      stencil.return %v : f64
    }
    stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
    func.return
  }

// CHECK:         func.func @stored_copy(%in : !stencil.field<[-4,68]xf64>, %midout : !stencil.field<[-4,68]xf64>, %out : !stencil.field<[-4,68]xf64>) {
// CHECK-NEXT:      %int = stencil.load %in : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %midt = stencil.apply(%inb = %int : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %inb[-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %midt to %midout(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
// CHECK-NEXT:      %outt = stencil.apply(%midb = %midt : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %v = stencil.access %midb[-1] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %v : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %outt to %out(<[0], [68]>) : !stencil.temp<?xf64> to !stencil.field<[-4,68]xf64>
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
      %10 = stencil.access %8[0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
      %11 = arith.addf %9, %10 : f64
      stencil.return %11 : f64
    }
    stencil.store %7 to %1(<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
    func.return
  }

// CHECK:         func.func @combine(%0 : !stencil.field<?x?xf64>) {
// CHECK-NEXT:      %1 = stencil.cast %0 : !stencil.field<?x?xf64> -> !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      %2 = stencil.apply() -> (!stencil.temp<[1,33]x[2,66]xf64>) {
// CHECK-NEXT:        %3 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        stencil.return %3 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %3 = stencil.apply() -> (!stencil.temp<[33,65]x[2,66]xf64>) {
// CHECK-NEXT:        %4 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        stencil.return %4 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = stencil.combine 0 at 33 lower = (%2 : !stencil.temp<[1,33]x[2,66]xf64>) upper = (%3 : !stencil.temp<[33,65]x[2,66]xf64>) : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:      %5 = stencil.buffer %4 : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:      %6 = stencil.apply(%7 = %5 : !stencil.temp<[1,65]x[2,66]xf64>) -> (!stencil.temp<[1,65]x[2,66]xf64>) {
// CHECK-NEXT:        %8 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %9 = stencil.access %7[0, 0] : !stencil.temp<[1,65]x[2,66]xf64>
// CHECK-NEXT:        %10 = arith.addf %8, %9 : f64
// CHECK-NEXT:        stencil.return %10 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %6 to %1(<[1, 2], [65, 66]>) : !stencil.temp<[1,65]x[2,66]xf64> to !stencil.field<[-3,67]x[-3,67]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}

// CHECK-NEXT: }

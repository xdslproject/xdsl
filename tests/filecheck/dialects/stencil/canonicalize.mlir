// RUN: xdsl-opt %s -p canonicalize --split-input-file | filecheck %s

func.func @dup_operand(%f : !stencil.field<[0,64]xf64>, %of1 : !stencil.field<[0,64]xf64>, %of2 : !stencil.field<[0,64]xf64>){
    %t = stencil.load %f : !stencil.field<[0,64]xf64> -> !stencil.temp<?xf64>
    %o1, %o2 = stencil.apply(%one = %t : !stencil.temp<?xf64>, %two = %t : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<?xf64>) {
        %1 = stencil.access %one[0] : !stencil.temp<?xf64>
        %2 = stencil.access %two[0] : !stencil.temp<?xf64>
        stencil.return %1, %2 : f64, f64
    }
    stencil.store %o1 to %of1(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
    stencil.store %o2 to %of2(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
    return
}

// CHECK:         func.func @dup_operand(%f : !stencil.field<[0,64]xf64>, %of1 : !stencil.field<[0,64]xf64>, %of2 : !stencil.field<[0,64]xf64>) {
// CHECK-NEXT:      %t = stencil.load %f : !stencil.field<[0,64]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %o1, %o2 = stencil.apply(%one = %t : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<?xf64>) {
// CHECK-NEXT:        %0 = stencil.access %one[0] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %0, %0 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %o1 to %of1(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
// CHECK-NEXT:      stencil.store %o2 to %of2(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// -----

func.func @unused_res(%f1 : !stencil.field<[0,64]xf64>, %f2 : !stencil.field<[0,64]xf64>, %of : !stencil.field<[0,64]xf64>){
    %t1 = stencil.load %f1 : !stencil.field<[0,64]xf64> -> !stencil.temp<?xf64>
    %t2 = stencil.load %f2 : !stencil.field<[0,64]xf64> -> !stencil.temp<?xf64>
    %o1, %o2 = stencil.apply(%one = %t1 : !stencil.temp<?xf64>, %two = %t2 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>, !stencil.temp<?xf64>) {
        %1 = stencil.access %one[0] : !stencil.temp<?xf64>
        %2 = stencil.access %two[0] : !stencil.temp<?xf64>
        stencil.return %1, %2 : f64, f64
    }
    stencil.store %o1 to %of(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
    return
}

// CHECK:         func.func @unused_res(%f1 : !stencil.field<[0,64]xf64>, %f2 : !stencil.field<[0,64]xf64>, %of : !stencil.field<[0,64]xf64>) {
// CHECK-NEXT:      %t1 = stencil.load %f1 : !stencil.field<[0,64]xf64> -> !stencil.temp<?xf64>
// CHECK-NEXT:      %o1 = stencil.apply(%one = %t1 : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
// CHECK-NEXT:        %0 = stencil.access %one[0] : !stencil.temp<?xf64>
// CHECK-NEXT:        stencil.return %0 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      stencil.store %o1 to %of(<[0], [64]>) : !stencil.temp<?xf64> to !stencil.field<[0,64]xf64>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// RUN: xdsl-opt -t csl %s

"memref.global"() {"sym_name" = "A", "type" = memref<24xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "x", "type" = memref<6xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "b", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()
"memref.global"() {"sym_name" = "y", "type" = memref<4xf32>, "sym_visibility" = "public", "initial_value" = dense<0> : tensor<1xindex>} : () -> ()

func.func @initialize() {
  %lb = arith.constant   0 : i16
  %ub = arith.constant  24 : i16
  %step = arith.constant 1 : i16

  %A = memref.get_global @A : memref<24xf32>
  %x = memref.get_global @x : memref<6xf32>
  %b = memref.get_global @b : memref<4xf32>
  %y = memref.get_global @y : memref<4xf32>

  scf.for %idx = %lb to %ub step %step {
    %idx_f32 = "arith.sitofp"(%idx) : (i16) -> f32
    %idx_index = "arith.index_cast"(%idx) : (i16) -> index
    memref.store %idx_f32, %A[%idx_index] : memref<24xf32>
  }

  %ub_6 = arith.constant 6 : i16

  scf.for %j = %lb to %ub step %step {
    %val = arith.constant 1.0 : f32
    %j_idx = "arith.index_cast"(%j) : (i16) -> index
    memref.store %val, %x[%j_idx] : memref<6xf32>
  }

  %ub_4 = arith.constant 6 : i16

  scf.for %i = %lb to %ub_4 step %step {
    %v2 = arith.constant 2.0 : f32
    %v0 = arith.constant 0.0 : f32
    %i_idx = "arith.index_cast"(%i) : (i16) -> index
    memref.store %v2, %b[%i_idx] : memref<4xf32>
    memref.store %v0, %y[%i_idx] : memref<4xf32>
  }

  func.return
}


// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "A", "sym_visibility" = "public", "type" = memref<24xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "x", "sym_visibility" = "public", "type" = memref<6xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "b", "sym_visibility" = "public", "type" = memref<4xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: //unknown op Global("memref.global"() <{"sym_name" = "y", "sym_visibility" = "public", "type" = memref<4xf32>, "initial_value" = dense<0> : tensor<1xindex>}> : () -> ())
// CHECK-NEXT: fn initialize() {
// CHECK-NEXT:   const lb : i16 = 0;
// CHECK-NEXT:   const ub : i16 = 24;
// CHECK-NEXT:   const step : i16 = 1;
// CHECK-NEXT:   //unknown op GetGlobal(%A = memref.get_global @A : memref<24xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%x = memref.get_global @x : memref<6xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%b = memref.get_global @b : memref<4xf32>)
// CHECK-NEXT:   //unknown op GetGlobal(%y = memref.get_global @y : memref<4xf32>)
// CHECK-NEXT:   //unknown op For(
// CHECK-NEXT:   //    scf.for %idx = %lb to %ub step %step : i16 {
// CHECK-NEXT:   //      %idx_f32 = "arith.sitofp"(%idx) : (i16) -> f32
// CHECK-NEXT:   //      %idx_index = "arith.index_cast"(%idx) : (i16) -> index
// CHECK-NEXT:   //      memref.store %idx_f32, %A[%idx_index] : memref<24xf32>
// CHECK-NEXT:   //    }
// CHECK-NEXT:   //)
// CHECK-NEXT:   const ub_6 : i16 = 6;
// CHECK-NEXT:   //unknown op For(
// CHECK-NEXT:   //    scf.for %j = %lb to %ub step %step : i16 {
// CHECK-NEXT:   //      %val = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   //      %j_idx = "arith.index_cast"(%j) : (i16) -> index
// CHECK-NEXT:   //      memref.store %val, %x[%j_idx] : memref<6xf32>
// CHECK-NEXT:   //    }
// CHECK-NEXT:   //)
// CHECK-NEXT:   const ub_4 : i16 = 6;
// CHECK-NEXT:   //unknown op For(
// CHECK-NEXT:   //    scf.for %i = %lb to %ub_4 step %step : i16 {
// CHECK-NEXT:   //      %v2 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   //      %v0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   //      %i_idx = "arith.index_cast"(%i) : (i16) -> index
// CHECK-NEXT:   //      memref.store %v2, %b[%i_idx] : memref<4xf32>
// CHECK-NEXT:   //      memref.store %v0, %y[%i_idx] : memref<4xf32>
// CHECK-NEXT:   //    }
// CHECK-NEXT:   //)
// CHECK-NEXT:   //unknown op Return(func.return)
// CHECK-NEXT: }

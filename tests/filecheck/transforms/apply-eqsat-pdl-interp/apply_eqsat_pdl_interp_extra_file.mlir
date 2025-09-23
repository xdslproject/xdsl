// RUN: xdsl-opt %s -p 'apply-eqsat-pdl-interp{pdl_interp_file="%p/extra_file.mlir"}' | filecheck %s

// CHECK:      %x_c = eqsat.eclass %x : i32
// CHECK-NEXT: %zero = arith.constant 0 : i32
// CHECK-NEXT: %a = arith.muli %x_c, %a_c : i32
// CHECK-NEXT: %a_c = eqsat.eclass %a, %zero, %b : i32
// CHECK-NEXT: %b = arith.subi %x_c, %x_c : i32
// CHECK-NEXT: func.return %a_c, %a_c : i32, i32

func.func @impl(%x: i32) -> (i32, i32) {
  %x_c = eqsat.eclass %x : i32

  %zero = arith.constant 0 : i32
  %zero_c = eqsat.eclass %zero : i32

  %a = arith.muli %x_c, %zero_c : i32
  %a_c = eqsat.eclass %a : i32

  %b = arith.subi %x_c, %x_c : i32
  %b_c = eqsat.eclass %b : i32

  
  func.return %a_c, %b_c : i32, i32
}

// RUN: xdsl-opt -p convert-scf-to-x86-scf %s | filecheck %s

// CHECK-LABEL:    func.func @nested(%src : index, %dst : index) {
//  CHECK-NEXT:      %zero_outer = arith.constant 0 : index
//  CHECK-NEXT:      %step_outer = arith.constant 4 : index
//  CHECK-NEXT:      %forty_outer = arith.constant 40 : index
//  CHECK-NEXT:      %zero_outer_1 = builtin.unrealized_conversion_cast %zero_outer : index to !x86.reg
//  CHECK-NEXT:      %forty_outer_1 = builtin.unrealized_conversion_cast %forty_outer : index to !x86.reg
//  CHECK-NEXT:      %step_outer_1 = builtin.unrealized_conversion_cast %step_outer : index to !x86.reg
//  CHECK-NEXT:      x86_scf.for %offset_outer : !x86.reg  = %zero_outer_1 to %forty_outer_1 step %step_outer_1 {
//  CHECK-NEXT:        %offset_outer_1 = builtin.unrealized_conversion_cast %offset_outer : !x86.reg to index
//  CHECK-NEXT:        %zero_inner = arith.constant 0 : index
//  CHECK-NEXT:        %step_inner = arith.constant 2 : index
//  CHECK-NEXT:        %forty_inner = arith.constant 40 : index
//  CHECK-NEXT:        %zero_inner_1 = builtin.unrealized_conversion_cast %zero_inner : index to !x86.reg
//  CHECK-NEXT:        %forty_inner_1 = builtin.unrealized_conversion_cast %forty_inner : index to !x86.reg
//  CHECK-NEXT:        %step_inner_1 = builtin.unrealized_conversion_cast %step_inner : index to !x86.reg
//  CHECK-NEXT:        x86_scf.for %offset_inner : !x86.reg  = %zero_inner_1 to %forty_inner_1 step %step_inner_1 {
//  CHECK-NEXT:          %offset_inner_1 = builtin.unrealized_conversion_cast %offset_inner : !x86.reg to index
//  CHECK-NEXT:          "test.op"(%src, %dst, %offset_outer_1, %offset_inner_1) : (index, index, index, index) -> ()
//  CHECK-NEXT:        }
//  CHECK-NEXT:      }
//  CHECK-NEXT:      func.return
//  CHECK-NEXT:    }
func.func @nested(%src: index, %dst: index) {
  %zero_outer = arith.constant 0 : index
  %step_outer = arith.constant 4 : index
  %forty_outer = arith.constant 40 : index
  scf.for %offset_outer = %zero_outer to %forty_outer step %step_outer {
    %zero_inner = arith.constant 0 : index
    %step_inner = arith.constant 2 : index
    %forty_inner = arith.constant 40 : index
    scf.for %offset_inner = %zero_inner to %forty_inner step %step_inner {
      "test.op"(%src, %dst, %offset_outer, %offset_inner) : (index, index, index, index) -> ()
    }
  }
  func.return
}

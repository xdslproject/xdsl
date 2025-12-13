// RUN: xdsl-opt %s -p apply-eqsat-pdl-interp | filecheck %s

func.func @impl() -> i32 {
  %two   = arith.constant 2  : i32
  %twoc = eqsat.eclass %two : i32

  %a   = arith.constant 5 : i32
  %ac = eqsat.eclass %a  : i32

  // a * 2
  %mul   = arith.muli %ac, %twoc : i32
  %mulc = eqsat.eclass %mul       : i32
  
  // (a * 2) / 2
  %div   = arith.divui %mulc, %twoc : i32
  %divc = eqsat.eclass %div : i32
  
  func.return %divc : i32
}

// CHECK:     func.func @impl() -> i32 {
// CHECK-NEXT:  %two = arith.constant 2 : i32
// CHECK-NEXT:  %twoc = eqsat.eclass %two : i32
// CHECK-NEXT:  %a = arith.constant 5 : i32
// CHECK-NEXT:  %mul = arith.muli %divc, %twoc : i32
// CHECK-NEXT:  %mulc = eqsat.eclass %mul : i32
// CHECK-NEXT:  %0 = arith.constant 1 : i32
// CHECK-NEXT:  %1 = eqsat.const_eclass %0, %2 (constant = 1 : i32) : i32
// CHECK-NEXT:  %2 = arith.divui %twoc, %twoc : i32
// CHECK-NEXT:  %3 = arith.muli %divc, %1 : i32
// CHECK-NEXT:  %div = arith.divui %mulc, %twoc : i32
// CHECK-NEXT:  %divc = eqsat.eclass %div, %3, %a : i32
// CHECK-NEXT:  func.return %divc : i32
// CHECK-NEXT: }

pdl_interp.func @matcher(%arg0: !pdl.operation) {
  %0 = eqsat_pdl_interp.get_result 0 of %arg0
  pdl_interp.is_not_null %0 : !pdl.value -> ^bb2, ^bb1
^bb1:  // 21 preds: ^bb0, ^bb2, ^bb3, ^bb4, ^bb5, ^bb7, ^bb8, ^bb19, ^bb20, ^bb21, ^bb22, ^bb23, ^bb24, ^bb25, ^bb26, ^bb27, ^bb28, ^bb29, ^bb30, ^bb31, ^bb32
  eqsat_pdl_interp.finalize
^bb2:  // pred: ^bb0
  // "pdl_interp.switch_operation_name"(%arg0)[^bb1, ^bb3, ^bb19] <{caseValues = ["arith.divui", "arith.muli"]}> : (!pdl.operation) -> ()
  pdl_interp.switch_operation_name of %arg0 to ["arith.divui", "arith.muli"](^bb3, ^bb19) -> ^bb1
^bb3:  // pred: ^bb2
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb4, ^bb1
^bb4:  // pred: ^bb3
  pdl_interp.check_result_count of %arg0 is 1 -> ^bb5, ^bb1
^bb5:  // pred: ^bb4
  %1 = pdl_interp.get_operand 0 of %arg0
  pdl_interp.is_not_null %1 : !pdl.value -> ^bb6, ^bb1
^bb6:  // pred: ^bb5
  %2 = pdl_interp.get_operand 1 of %arg0
  pdl_interp.is_not_null %2 : !pdl.value -> ^bb9, ^bb7
^bb7:  // 11 preds: ^bb6, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18
  %3 = pdl_interp.get_operand 1 of %arg0
  pdl_interp.are_equal %1, %3 : !pdl.value -> ^bb8, ^bb1
^bb8:  // pred: ^bb7
  %4 = pdl_interp.get_value_type of %0 : !pdl.type
  pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%4, %arg0 : !pdl.type, !pdl.operation) : benefit(1), generatedOps(["arith.constant"]), loc([%arg0]), root("arith.divui") -> ^bb1
^bb9:  // pred: ^bb6
  %5 = eqsat_pdl_interp.get_defining_op of %1 : !pdl.value
  pdl_interp.is_not_null %5 : !pdl.operation -> ^bb10, ^bb7
^bb10:  // pred: ^bb9
  pdl_interp.check_operation_name of %5 is "arith.muli" -> ^bb11, ^bb7
^bb11:  // pred: ^bb10
  pdl_interp.check_operand_count of %5 is 2 -> ^bb12, ^bb7
^bb12:  // pred: ^bb11
  pdl_interp.check_result_count of %5 is 1 -> ^bb13, ^bb7
^bb13:  // pred: ^bb12
  %6 = pdl_interp.get_operand 0 of %5
  pdl_interp.is_not_null %6 : !pdl.value -> ^bb14, ^bb7
^bb14:  // pred: ^bb13
  %7 = pdl_interp.get_operand 1 of %5
  pdl_interp.is_not_null %7 : !pdl.value -> ^bb15, ^bb7
^bb15:  // pred: ^bb14
  %8 = eqsat_pdl_interp.get_result 0 of %5
  pdl_interp.is_not_null %8 : !pdl.value -> ^bb16, ^bb7
^bb16:  // pred: ^bb15
  pdl_interp.are_equal %8, %1 : !pdl.value -> ^bb17, ^bb7
^bb17:  // pred: ^bb16
  %9 = pdl_interp.get_value_type of %8 : !pdl.type
  %10 = pdl_interp.get_value_type of %0 : !pdl.type
  pdl_interp.are_equal %9, %10 : !pdl.type -> ^bb18, ^bb7
^bb18:  // pred: ^bb17
  eqsat_pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%7, %2, %9, %6, %arg0 : !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) : benefit(1), generatedOps(["arith.divui", "arith.muli"]), loc([%arg0, %5]), root("arith.divui") -> ^bb7
^bb19:  // pred: ^bb2
  pdl_interp.check_operand_count of %arg0 is 2 -> ^bb20, ^bb1
^bb20:  // pred: ^bb19
  pdl_interp.check_result_count of %arg0 is 1 -> ^bb21, ^bb1
^bb21:  // pred: ^bb20
  %11 = pdl_interp.get_operand 0 of %arg0
  pdl_interp.is_not_null %11 : !pdl.value -> ^bb22, ^bb1
^bb22:  // pred: ^bb21
  %12 = pdl_interp.get_operand 1 of %arg0
  pdl_interp.is_not_null %12 : !pdl.value -> ^bb23, ^bb1
^bb23:  // pred: ^bb22
  %13 = eqsat_pdl_interp.get_defining_op of %12 : !pdl.value
  pdl_interp.is_not_null %13 : !pdl.operation -> ^bb24, ^bb1
^bb24:  // pred: ^bb23
  pdl_interp.check_operation_name of %13 is "arith.constant" -> ^bb25, ^bb1
^bb25:  // pred: ^bb24
  pdl_interp.check_operand_count of %13 is 0 -> ^bb26, ^bb1
^bb26:  // pred: ^bb25
  pdl_interp.check_result_count of %13 is 1 -> ^bb27, ^bb1
^bb27:  // pred: ^bb26
  %14 = pdl_interp.get_attribute "value" of %13
  pdl_interp.is_not_null %14 : !pdl.attribute -> ^bb28, ^bb1
^bb28:  // pred: ^bb27
  pdl_interp.check_attribute %14 is 1 : i32 -> ^bb29, ^bb1
^bb29:  // pred: ^bb28
  %15 = eqsat_pdl_interp.get_result 0 of %13
  pdl_interp.is_not_null %15 : !pdl.value -> ^bb30, ^bb1
^bb30:  // pred: ^bb29
  pdl_interp.are_equal %15, %12 : !pdl.value -> ^bb31, ^bb1
^bb31:  // pred: ^bb30
  %16 = pdl_interp.get_value_type of %15 : !pdl.type
  %17 = pdl_interp.get_value_type of %0 : !pdl.type
  pdl_interp.are_equal %16, %17 : !pdl.type -> ^bb32, ^bb1
^bb32:  // pred: ^bb31
  eqsat_pdl_interp.record_match @rewriters::@pdl_generated_rewriter_1(%11, %arg0 : !pdl.value, !pdl.operation) : benefit(1), loc([%arg0, %13]), root("arith.muli") -> ^bb1
}
module @rewriters {
  pdl_interp.func @pdl_generated_rewriter(%arg0: !pdl.type, %arg1: !pdl.operation) {
    %0 = pdl_interp.create_attribute 1 : i32
    %1 = eqsat_pdl_interp.create_operation "arith.constant" {"value" = %0}  -> (%arg0 : !pdl.type)
    %2 = eqsat_pdl_interp.get_results of %1 : !pdl.range<value>
    eqsat_pdl_interp.replace %arg1 with (%2 : !pdl.range<value>)
    eqsat_pdl_interp.finalize
  }
  pdl_interp.func @pdl_generated_rewriter_0(%arg0: !pdl.value, %arg1: !pdl.value, %arg2: !pdl.type, %arg3: !pdl.value, %arg4: !pdl.operation) {
    %0 = eqsat_pdl_interp.create_operation "arith.divui"(%arg0, %arg1 : !pdl.value, !pdl.value)  -> (%arg2 : !pdl.type)
    %1 = eqsat_pdl_interp.get_result 0 of %0
    %2 = eqsat_pdl_interp.create_operation "arith.muli"(%arg3, %1 : !pdl.value, !pdl.value)  -> (%arg2 : !pdl.type)
    %3 = eqsat_pdl_interp.get_result 0 of %2
    %4 = eqsat_pdl_interp.get_results of %2 : !pdl.range<value>
    eqsat_pdl_interp.replace %arg4 with (%4 : !pdl.range<value>)
    eqsat_pdl_interp.finalize
  }
  pdl_interp.func @pdl_generated_rewriter_1(%arg0: !pdl.value, %arg1: !pdl.operation) {
    eqsat_pdl_interp.replace %arg1 with (%arg0 : !pdl.value)
    eqsat_pdl_interp.finalize
  }
}

// // (x * y) / z -> x * (y/z)
// pdl.pattern : benefit(1) {
//   %x = pdl.operand
//   %y = pdl.operand
//   %z = pdl.operand
//   %type = pdl.type
//   %mulop = pdl.operation "arith.muli" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   %mul = pdl.result 0 of %mulop
//   %resultop = pdl.operation "arith.divui" (%mul, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   %result = pdl.result 0 of %resultop
//   pdl.rewrite %resultop {
//     %newdivop = pdl.operation "arith.divui" (%y, %z : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//     %newdiv = pdl.result 0 of %newdivop
//     %newresultop = pdl.operation "arith.muli" (%x, %newdiv : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//     %newresult = pdl.result 0 of %newresultop
//     pdl.replace %resultop with %newresultop
//   }
// }

// // x / x -> 1
// pdl.pattern : benefit(1) {
//   %x = pdl.operand
//   %type = pdl.type
//   %resultop = pdl.operation "arith.divui" (%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   pdl.rewrite %resultop {
//     %2 = pdl.attribute = 1 : i32
//     %3 = pdl.operation "arith.constant" {"value" = %2} -> (%type : !pdl.type)
//     pdl.replace %resultop with %3
//   }
// }

// // x * 1 -> x
// pdl.pattern : benefit(1) {
//   %x = pdl.operand
//   %type = pdl.type
//   %one = pdl.attribute = 1 : i32
//   %constop = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
//   %const = pdl.result 0 of %constop
//   %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
//   pdl.rewrite %mulop {
//     pdl.replace %mulop with (%x : !pdl.value)
//   }
// }

// RUN:xdsl-opt %s --split-input-file -p 'apply-individual-rewrite{matched_operation_index=2 operation_name="arith.addi" pattern_name="AdditionOfSameVariablesToMultiplyByTwo"}'| filecheck %s


// CHECK:      %v = "test.op"() : () -> i32
// CHECK-NEXT: %[[#two:]] = arith.constant 2 : i32
// CHECK-NEXT: %{{.*}} = arith.muli %v, %[[#two]] : i32

%v = "test.op"() : () -> (i32)
%1 = arith.addi %v, %v : i32

// -----

// CHECK:      %v = "test.op"() : () -> i1
// CHECK-NEXT: %[[#zero:]] = arith.constant false
// CHECK-NEXT: %{{.*}} = arith.muli %v, %[[#zero]] : i1

%v = "test.op"() : () -> (i1)
%1 = arith.addi %v, %v : i1

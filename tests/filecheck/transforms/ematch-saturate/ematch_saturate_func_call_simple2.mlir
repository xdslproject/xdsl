func.func @collatz_steps(%arg0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %1:3 = scf.while (%arg1 = %c0_i64, %arg2 = %arg0) : (i64, i64) -> (i64, i64, i64) {
      %2 = arith.cmpi ne, %arg2, %c1_i64 : i64
      %3:2 = scf.if %2 -> (i64, i64) {
        %4 = arith.remsi %arg2, %c2_i64 : i64
        %5 = arith.cmpi eq, %4, %c0_i64 : i64
        %6 = scf.if %5 -> (i64) {
          %8 = arith.divsi %arg2, %c2_i64 : i64
          scf.yield %8 : i64
        } else {
          %8 = arith.muli %arg2, %c3_i64 : i64
          %9 = arith.addi %8, %c1_i64 : i64
          scf.yield %9 : i64
        }
        %7 = arith.addi %arg1, %c1_i64 : i64
        scf.yield %7, %6 : i64, i64
      } else {
        scf.yield %0, %0 : i64, i64
      }
      scf.condition(%2) %3#0, %3#1, %arg1 : i64, i64, i64
    } do {
    ^bb0(%arg1: i64, %arg2: i64, %arg3: i64):
      scf.yield %arg1, %arg2 : i64, i64
    }
    return %1#2 : i64
  }
  func.func @max_collatz(%arg0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %1:4 = scf.while (%arg1 = %c1_i64, %arg2 = %c0_i64, %arg3 = %c0_i64) : (i64, i64, i64) -> (i64, i64, i64, i64) {
      %2 = arith.cmpi slt, %arg1, %arg0 : i64
      %3:3 = scf.if %2 -> (i64, i64, i64) {
        %4 = func.call @collatz_steps(%arg1) : (i64) -> i64
        %5 = arith.cmpi sgt, %4, %arg3 : i64
        %6 = arith.select %5, %arg1, %arg2 : i64
        %7 = arith.select %5, %4, %arg3 : i64
        %8 = arith.addi %arg1, %c1_i64 : i64
        scf.yield %8, %6, %7 : i64, i64, i64
      } else {
        scf.yield %0, %0, %0 : i64, i64, i64
      }
      scf.condition(%2) %3#0, %3#1, %3#2, %arg2 : i64, i64, i64, i64
    } do {
    ^bb0(%arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64):
      scf.yield %arg1, %arg2, %arg3 : i64, i64, i64
    }
    return %1#3 : i64
  }



  pdl_interp.func @matcher(%arg0 : !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "arith.remsi" -> ^bb0, ^bb30
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb2, ^bb1
  ^bb2:
    %0 = pdl_interp.get_operand 1 of %arg0
    pdl_interp.is_not_null %0 : !pdl.value -> ^bb3, ^bb1
  ^bb3:
    %1 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %1 : !pdl.value -> ^bb4, ^bb1
  ^bb4:
    %2 = ematch.get_class_result %1
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5, ^bb1
  ^bb5:
    %3 = ematch.get_class_vals %0
    pdl_interp.foreach %4 : !pdl.value in %3 {
      %5 = pdl_interp.get_defining_op of %4 : !pdl.value
      pdl_interp.is_not_null %5 : !pdl.operation -> ^bb6, ^bb7
    ^bb7:
      pdl_interp.continue
    ^bb6:
      pdl_interp.check_operation_name of %5 is "arith.constant" -> ^bb8, ^bb7
    ^bb8:
      pdl_interp.check_operand_count of %5 is 0 -> ^bb9, ^bb7
    ^bb9:
      pdl_interp.check_result_count of %5 is 1 -> ^bb10, ^bb7
    ^bb10:
      %6 = pdl_interp.get_attribute "value" of %5
      pdl_interp.is_not_null %6 : !pdl.attribute -> ^bb11, ^bb7
    ^bb11:
      %7 = pdl_interp.create_attribute 2 : i64
      pdl_interp.are_equal %6, %7 : !pdl.attribute -> ^bb12, ^bb7
    ^bb12:
      %8 = pdl_interp.get_result 0 of %5
      pdl_interp.is_not_null %8 : !pdl.value -> ^bb13, ^bb7
    ^bb13:
      %9 = ematch.get_class_result %8
      pdl_interp.is_not_null %9 : !pdl.value -> ^bb14, ^bb7
    ^bb14:
      pdl_interp.are_equal %9, %0 : !pdl.value -> ^bb15, ^bb7
    ^bb15:
      %10 = pdl_interp.get_value_type of %2 : !pdl.type
      pdl_interp.record_match @rewriters::@rem_by_2_to_and_1(%arg0, %10 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("arith.remui") -> ^bb7
  } -> ^bb30
  ^bb30:
    pdl_interp.check_operation_name of %arg0 is "arith.divsi" -> ^bb31, ^bb60
  ^bb31:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb32, ^bb1
  ^bb32:
    %30 = pdl_interp.get_operand 1 of %arg0
    pdl_interp.is_not_null %30 : !pdl.value -> ^bb33, ^bb1
  ^bb33:
    %31 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %31 : !pdl.value -> ^bb34, ^bb1
  ^bb34:
    %32 = ematch.get_class_result %31
    pdl_interp.is_not_null %32 : !pdl.value -> ^bb35, ^bb1
  ^bb35:
    %33 = ematch.get_class_vals %30
    pdl_interp.foreach %34 : !pdl.value in %33 {
      %35 = pdl_interp.get_defining_op of %34 : !pdl.value
      pdl_interp.is_not_null %35 : !pdl.operation -> ^bb36, ^bb37
    ^bb37:
      pdl_interp.continue
    ^bb36:
      pdl_interp.check_operation_name of %35 is "arith.constant" -> ^bb38, ^bb37
    ^bb38:
      pdl_interp.check_operand_count of %35 is 0 -> ^bb39, ^bb37
    ^bb39:
      pdl_interp.check_result_count of %35 is 1 -> ^bb40, ^bb37
    ^bb40:
      %36 = pdl_interp.get_attribute "value" of %35
      pdl_interp.is_not_null %36 : !pdl.attribute -> ^bb41, ^bb37
    ^bb41:
      %37 = pdl_interp.create_attribute 2 : i64
      pdl_interp.are_equal %36, %37 : !pdl.attribute -> ^bb42, ^bb37
    ^bb42:
      %38 = pdl_interp.get_result 0 of %35
      pdl_interp.is_not_null %38 : !pdl.value -> ^bb43, ^bb37
    ^bb43:
      %39 = ematch.get_class_result %38
      pdl_interp.is_not_null %39 : !pdl.value -> ^bb44, ^bb37
    ^bb44:
      pdl_interp.are_equal %39, %30 : !pdl.value -> ^bb45, ^bb37
    ^bb45:
      %40 = pdl_interp.get_value_type of %32 : !pdl.type
      pdl_interp.record_match @rewriters::@div_by_2_to_shr_1(%arg0, %40 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("arith.remui") -> ^bb37
  } -> ^bb60
  ^bb60:
    pdl_interp.check_operation_name of %arg0 is "arith.addi" -> ^bb61, ^bb90
  ^bb61:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb62, ^bb90
  ^bb62:
    %60 = pdl_interp.get_operand 0 of %arg0
    pdl_interp.is_not_null %60 : !pdl.value -> ^bb63, ^bb90
  ^bb63:
    %61 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %61 : !pdl.value -> ^bb64, ^bb90
  ^bb64:
    %62 = ematch.get_class_result %61
    pdl_interp.is_not_null %62 : !pdl.value -> ^bb65, ^bb90
  ^bb65:
    %63 = ematch.get_class_vals %60
    pdl_interp.foreach %64 : !pdl.value in %63 {
        %65 = pdl_interp.get_defining_op of %64 : !pdl.value
        pdl_interp.is_not_null %65 : !pdl.operation -> ^bb68, ^bb67
      ^bb67:
        pdl_interp.continue
      ^bb68:
        pdl_interp.check_operation_name of %65 is "arith.muli" -> ^bb69, ^bb67
      ^bb69:
        pdl_interp.check_operand_count of %65 is 2 -> ^bb70, ^bb67
      ^bb70:
        %66 = pdl_interp.get_operand 1 of %65
        pdl_interp.is_not_null %66 : !pdl.value -> ^bb72, ^bb67
      ^bb72:
        %68 = ematch.get_class_result %66
        pdl_interp.is_not_null %68 : !pdl.value -> ^bb73, ^bb67
      ^bb73:
        %69 = ematch.get_class_vals %66
        pdl_interp.foreach %70 : !pdl.value in %69 {
            %71 = pdl_interp.get_defining_op of %70 : !pdl.value
            pdl_interp.is_not_null %71 : !pdl.operation -> ^bb75, ^bb74
          ^bb74:
            pdl_interp.continue
          ^bb75:
            pdl_interp.check_operation_name of %71 is "arith.constant" -> ^bb76, ^bb74
          ^bb76:
            %72 = pdl_interp.get_attribute "value" of %71
            pdl_interp.is_not_null %72 : !pdl.attribute -> ^bb77, ^bb74
          ^bb77:
            %73 = pdl_interp.create_attribute 3 : i64
            pdl_interp.are_equal %72, %73 : !pdl.attribute -> ^bb78, ^bb74
          ^bb78:
            %74 = pdl_interp.get_result 0 of %71
            pdl_interp.is_not_null %74 : !pdl.value -> ^bb79, ^bb74
          ^bb79:
            %75 = ematch.get_class_result %74
            pdl_interp.is_not_null %75 : !pdl.value -> ^bb80, ^bb74
          ^bb80:
            pdl_interp.are_equal %75, %66 : !pdl.value -> ^bb81, ^bb74
          ^bb81:
            %76 = pdl_interp.get_value_type of %68 : !pdl.type
            pdl_interp.record_match @rewriters::@r3n1(%arg0, %65, %76 : !pdl.operation, !pdl.operation, !pdl.type) : benefit(1), loc([]), root("arith.muli") -> ^bb74
        } -> ^bb67
    } -> ^bb90
  ^bb90:
    pdl_interp.check_operation_name of %arg0 is "scf.if" -> ^bb110, ^bb1130
  ^bb111:
    pdl_interp.finalize
  ^bb110:
    pdl_interp.check_result_count of %arg0 is 1 -> ^bb112, ^bb111
  ^bb112:
    %100 = pdl_interp.get_operand 0 of %arg0
    pdl_interp.is_not_null %100 : !pdl.value -> ^bb113, ^bb111
  ^bb113:
    %101 = pdl_interp.get_result 0 of %arg0
    pdl_interp.is_not_null %101 : !pdl.value -> ^bb114, ^bb111
  ^bb114:
    %102 = ematch.get_class_result %101
    pdl_interp.is_not_null %102 : !pdl.value -> ^bb115, ^bb111
  ^bb115:
    %103 = ematch.get_class_vals %100
    pdl_interp.foreach %104 : !pdl.value in %103 {
      %105 = pdl_interp.get_defining_op of %104 : !pdl.value
      pdl_interp.is_not_null %105 : !pdl.operation -> ^bb116, ^bb117
    ^bb117:
      pdl_interp.continue
    ^bb116:
      pdl_interp.check_operation_name of %105 is "arith.constant" -> ^bb118, ^bb117
    ^bb118:
      pdl_interp.check_operand_count of %105 is 0 -> ^bb119, ^bb117
    ^bb119:
      pdl_interp.check_result_count of %105 is 1 -> ^bb1110, ^bb117
    ^bb1110:
      %106 = pdl_interp.get_attribute "value" of %105
      pdl_interp.is_not_null %106 : !pdl.attribute -> ^bb1111, ^bb117
    ^bb1111:
      %107 = pdl_interp.create_attribute 1 : i1
      pdl_interp.are_equal %106, %107 : !pdl.attribute -> ^bb1112, ^bb1116
    ^bb1112:
      %108 = pdl_interp.get_result 0 of %105
      pdl_interp.is_not_null %108 : !pdl.value -> ^bb1113, ^bb117
    ^bb1113:
      %109 = ematch.get_class_result %108
      pdl_interp.is_not_null %109 : !pdl.value -> ^bb1114, ^bb117
    ^bb1114:
      pdl_interp.are_equal %109, %100 : !pdl.value -> ^bb1115, ^bb117
    ^bb1115:
      %1010 = pdl_interp.get_value_type of %102 : !pdl.type
      pdl_interp.record_match @rewriters::@if_true_rewriter(%arg0, %1010 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb117
    ^bb1116:
      %1011 = pdl_interp.create_attribute 0 : i1
      pdl_interp.are_equal %106, %1011 : !pdl.attribute -> ^bb1117, ^bb117
    ^bb1117:
      %1012 = pdl_interp.get_result 0 of %105
      pdl_interp.is_not_null %1012 : !pdl.value -> ^bb1118, ^bb117
    ^bb1118:
      %1013 = ematch.get_class_result %1012
      pdl_interp.is_not_null %1013 : !pdl.value -> ^bb1119, ^bb117
    ^bb1119:
      pdl_interp.are_equal %1013, %100 : !pdl.value -> ^bb1120, ^bb117
    ^bb1120:
      %1014 = pdl_interp.get_value_type of %102 : !pdl.type
      pdl_interp.record_match @rewriters::@if_false_rewriter(%arg0, %1014 : !pdl.operation, !pdl.type) : benefit(1), loc([]), root("scf.if") -> ^bb117
    } -> ^bb111
    ^bb1130:
      pdl_interp.check_operation_name of %arg0 is "scf.execute_region" -> ^bb1131, ^bb1140
    ^bb1131:
      pdl_interp.record_match @rewriters::@execute_region_rewriter(%arg0 : !pdl.operation) : benefit(1) -> ^bb111
    ^bb1140:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb1141, ^bb111
    ^bb1141:
      %10100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb1142, ^bb111
    ^bb1142:
      %10101 = pdl_interp_region.get_region 0 of %10100 : !pdl_region.region
      %10102 = pdl_interp.apply_constraint "replace_return_with_yield"(%10101 : !pdl_region.region) : !pdl_region.region -> ^bb1143, ^bb111
    ^bb1143:
      %10103 = pdl_interp.apply_constraint "get_arguments_of_function"(%10100 : !pdl.operation) : !pdl.range<value> -> ^bb1144, ^bb111
    ^bb1144:
      %10104 = pdl_interp.get_result 0 of %arg0
      %10105 = pdl_interp.get_value_type of %10104 : !pdl.type
      %10106 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%10102 : !pdl_region.region) -> (%10105 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%10106, %10100, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb1145, ^bb111
    ^bb1145:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %10106 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb111
  }

  builtin.module @rewriters {
    pdl_interp.func @rem_by_2_to_and_1(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp.create_attribute 1 : i64
      %1 = pdl_interp.create_operation "arith.constant" {"value" = %0 } -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1

      %3 = pdl_interp.get_operand 0 of %arg0

      %4 = pdl_interp.create_operation "arith.andi"(%3, %2 : !pdl.value, !pdl.value) -> (%arg1 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = ematch.get_class_result %5

      %7 = pdl_interp.create_range %6 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %7: !pdl.range<value>

      pdl_interp.finalize
    }

    pdl_interp.func @div_by_2_to_shr_1(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp.create_attribute 1 : i64
      %1 = pdl_interp.create_operation "arith.constant" {"value" = %0} -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1

      %3 = pdl_interp.get_operand 0 of %arg0

      %4 = pdl_interp.create_operation "arith.shrui"(%3, %2 : !pdl.value, !pdl.value) -> (%arg1 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = ematch.get_class_result %5

      %7 = pdl_interp.create_range %6 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %7: !pdl.range<value>

      pdl_interp.finalize
    }

    pdl_interp.func @r3n1(%arg0 : !pdl.operation, %arg1 : !pdl.operation, %arg2 : !pdl.type) {
      %0 = pdl_interp.create_attribute 1 : i64
      %1 = pdl_interp.create_operation "arith.constant" {"value" = %0} -> (%arg2 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1

      %30 = pdl_interp.get_operand 0 of %arg1

      %3 = pdl_interp.create_operation "arith.shli"(%30, %2 : !pdl.value, !pdl.value) {"equivalence.cost" = %0} -> (%arg2 : !pdl.type)
      %40 = pdl_interp.get_result 0 of %3

      %4 = pdl_interp.create_operation "arith.addi"(%30, %40 : !pdl.value, !pdl.value) {"equivalence.cost" = %0} -> (%arg2 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4

      %6 = pdl_interp.create_operation "arith.addi"(%5, %2 : !pdl.value, !pdl.value) {"equivalence.cost" = %0} -> (%arg2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = ematch.get_class_result %7

      %a = pdl_interp.get_operand 0 of %arg0
      %b = pdl_interp.get_operand 1 of %arg0
      %d = pdl_interp.create_attribute 10 : i64
      %c = pdl_interp.create_operation "arith.muli"(%a, %b : !pdl.value, !pdl.value) {"equivalence.cost" = %d} -> (%arg2 : !pdl.type)

      %e = pdl_interp.get_result 0 of %c
      pdl_interp.replace %arg0 with (%e : !pdl.value)

      %9 = pdl_interp.create_range %8 : !pdl.value
      ematch.union %c : !pdl.operation, %9 : !pdl.range<value>

      pdl_interp.finalize
    }

    pdl_interp.func @if_true_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

     pdl_interp.func @if_false_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.type) {
      %0 = pdl_interp_region.get_region 1 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%0 : !pdl_region.region) -> (%arg1 : !pdl.type)
      %2 = pdl_interp.get_result 0 of %1
      %3 = ematch.get_class_result %2
      %4 = pdl_interp.create_range %3 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %4 : !pdl.range<value>
      pdl_interp.finalize
    }

    pdl_interp.func @execute_region_rewriter(%arg0: !pdl.operation) {
      %0 = pdl_interp_region.get_region 0 of %arg0 : !pdl_region.region
      %1 = pdl_interp_region.inline_region %arg0 with (%0 : !pdl_region.region)
      pdl_interp.replace %arg0 with (%1 : !pdl.value)
      pdl_interp.finalize
    }

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }
  }
func.func private @g() -> i32
func.func private @h(%arg0: i32) -> i32

func.func @f() -> i32 {
    %res = equivalence.graph : () -> i32 {
        // x' = g()
        %x_2 = func.call @g() : () -> i32

        // a = E-class(x')
        %a = equivalence.class %x_2 : i32

        // b = h(a)
        %b = func.call @h(%a) : (i32) -> i32

        equivalence.yield %b : i32
    }

    func.return %res : i32
}

func.func @main() -> i32 {
    %res_1 = equivalence.graph : () -> i32 {
        // x = g()
        %x = func.call @g() : () -> i32

        // r = call f(...)
        %r = func.call @f() : () -> i32

        // y = E-class(x)
        %y = equivalence.class %x : i32

        equivalence.yield %y : i32
    }

    return %res_1 : i32
}


pdl_interp.func @matcher(%arg0 : !pdl.operation) {
    pdl_interp.check_operation_name of %arg0 is "execute_region" -> ^bb40, ^bb1
  ^bb1:
    pdl_interp.finalize
  ^bb40:
      pdl_interp.check_operation_name of %arg0 is "func.call" -> ^bb41, ^bb1
    ^bb41:
      pdl_interp.check_result_count of %arg0 is 1 -> ^bb100, ^bb1
    ^bb100:
      %100 = pdl_interp.apply_constraint "get_function_call"(%arg0 : !pdl.operation) : !pdl.operation -> ^bb42, ^bb1
    ^bb42:
      %110 = pdl_interp.get_attribute "sym_visibility" of %100
      %112 = pdl_interp.create_attribute "private"
      pdl_interp.are_equal %110, %112 : !pdl.attribute -> ^bb1, ^bb48
    ^bb48:
      %101 = pdl_interp_region.get_region 0 of %100 : !pdl_region.region
      %1011 = pdl_interp_region.get_operation 0 of %101
      %102 = pdl_interp.apply_constraint "replace_return_with_yield"(%101 : !pdl_region.region) : !pdl_region.region -> ^bb43, ^bb1
    ^bb43:
      %103 = pdl_interp.apply_constraint "get_arguments_of_function"(%100 : !pdl.operation) : !pdl.range<value> -> ^bb46, ^bb1
    ^bb46:
      %104 = pdl_interp.get_result 0 of %arg0
      %105 = pdl_interp.get_value_type of %104 : !pdl.type
      %106 = pdl_interp_region.create_operation_with_region "scf.execute_region"(%102 : !pdl_region.region) -> (%105 : !pdl.type)
      pdl_interp.apply_constraint "replace_func_args_with_correct_definitions"(%106, %100, %arg0 : !pdl.operation, !pdl.operation, !pdl.operation) -> ^bb45, ^bb1
    ^bb45:
      pdl_interp.record_match @rewriters::@func_call_rewriter(%arg0, %106 : !pdl.operation, !pdl.operation) : benefit(1) -> ^bb1
      }

      builtin.module @rewriters {

    pdl_interp.func @func_call_rewriter(%arg0 : !pdl.operation, %arg1 : !pdl.operation) {
      %0 = pdl_interp.get_result 0 of %arg1
      %1 = ematch.get_class_result %0
      %2 = pdl_interp.create_range %1 : !pdl.value
      ematch.union %arg0 : !pdl.operation, %2 : !pdl.range<value>
      pdl_interp.finalize
    }
  }
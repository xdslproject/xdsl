// RUN: irdl-to-pyrdl %s | filecheck %s
// RUN: irdl-to-pyrdl %s -o %t.py && cat %t.py | filecheck %s
// RUN: irdl-to-pyrdl %s -o %t.py && python %t.py

// CHECK:      from xdsl.irdl import *
// CHECK-NEXT: from xdsl.ir import *

builtin.module {
  irdl.dialect @testd {

    irdl.type @parametric {
      %0 = irdl.any
      irdl.parameters(elem: %0)
    }
// CHECK:      @irdl_attr_definition
// CHECK-NEXT: class parametric(ParametrizedAttribute, TypeAttribute):
// CHECK-NEXT:     name = "testd.parametric"
// CHECK-NEXT:     elem: Attribute


    irdl.attribute @parametric_attr {
      %0 = irdl.any
      irdl.parameters(elem: %0)
    }
// CHECK:      @irdl_attr_definition
// CHECK-NEXT: class parametric_attr(ParametrizedAttribute):
// CHECK-NEXT:     name = "testd.parametric_attr"
// CHECK-NEXT:     elem: Attribute


    irdl.type @attr_in_type_out {
      %0 = irdl.any
      irdl.parameters(param: %0)
    }
// CHECK:      @irdl_attr_definition
// CHECK-NEXT: class attr_in_type_out(ParametrizedAttribute, TypeAttribute):
// CHECK-NEXT:     name = "testd.attr_in_type_out"
// CHECK-NEXT:     param: Attribute



    irdl.operation @my_eq {
      %0 = irdl.is i32
      irdl.results(out: %0)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class MyEqOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.my_eq"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()


    irdl.operation @any_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(out: %2)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class AnyOfOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.any_of"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()


    irdl.operation @all_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.all_of(%2, %1)
      irdl.results(out: %3)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class AllOfOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.all_of"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()


    irdl.operation @any {
      %0 = irdl.any
      irdl.results(out: %0)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class AnyOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.any"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()


    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @testd::@parametric<%0>
      irdl.results(out: %1)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class DynbaseOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.dynbase"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()

    irdl.operation @dynparams {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @testd::@parametric<%2>
      irdl.results(out: %3)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class DynparamsOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.dynparams"
// CHECK-NEXT:     out = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()

    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(out1: %2, out2: %2)
    }
// CHECK:      @irdl_op_definition
// CHECK-NEXT: class ConstraintVarsOp(IRDLOperation):
// CHECK-NEXT:     name = "testd.constraint_vars"
// CHECK-NEXT:     out1 = result_def()
// CHECK-NEXT:     out2 = result_def()
// CHECK-NEXT:     regs = var_region_def()
// CHECK-NEXT:     succs = var_successor_def()
  }
}

// CHECK: testd = Dialect("testd", [MyEqOp, AnyOfOp, AllOfOp, AnyOp, DynbaseOp, DynparamsOp, ConstraintVarsOp], [parametric, parametric_attr, attr_in_type_out])

// RUN: xdsl-opt -p eqsat-serialize-egraph %s | filecheck %s

// CHECK: {"nodes": {"enode_3": {"op": "arith.addi", "eclass": "eclass_1", "children": ["enode_1", "enode_2"]}, "enode_1": {"op": "arg 0", "eclass": "eclass_1", "children": []}, "enode_2": {"op": "arith.constant 0", "eclass": "eclass_2", "children": []}}}
func.func @egraph(%a : i64) {
  %zero = arith.constant 0 : i64
  %c_res = eqsat.eclass %a, %sum : i64
  %c_zero = eqsat.eclass %zero : i64
  %sum = arith.addi %c_res, %c_zero : i64
  func.return
}

// RUN: xdsl-opt -p eqsat-serialize-egraph %s | filecheck %s

// CHECK: {"nodes": {"enode_0": {"op": "arg 0", "eclass": "eclass_0", "children": []}, "enode_1": {"op": "arith.constant 1", "eclass": "eclass_1", "children": []}, "enode_2": {"op": "arith.constant 2", "eclass": "eclass_2", "children": []}, "enode_3": {"op": "arith.shli", "eclass": "eclass_3", "children": ["enode_0", "enode_1"]}, "enode_4": {"op": "arith.muli", "eclass": "eclass_3", "children": ["enode_0", "enode_2"]}}}
func.func @egraph(%a : index, %b : index) -> index {
  %a_eq = eqsat.eclass %a : index
  %one = arith.constant 1 : index
  %one_eq = eqsat.eclass %one : index
  %two = arith.constant 2 : index
  %two_eq = eqsat.eclass %two : index
  %a_shift_one = arith.shli %a_eq, %one_eq : index
  %a_times_two = arith.muli %a_eq, %two_eq : index
  %res_eq = eqsat.eclass %a_shift_one, %a_times_two : index
  func.return %res_eq : index
}

// RUN: xdsl-opt -p eqsat-serialize-egraph %s | filecheck %s

// CHECK: {"nodes": {"enode_3": {"op": "arith.muli", "eclass": "eclass_1", "children": ["enode_1", "enode_2"]}, "enode_5": {"op": "arith.shli", "eclass": "eclass_1", "children": ["enode_1", "enode_4"]}, "enode_2": {"op": "arith.constant 2", "eclass": "eclass_2", "children": []}, "enode_4": {"op": "arith.constant 1", "eclass": "eclass_3", "children": []}, "enode_1": {"op": "arg 0", "eclass": "eclass_4", "children": []}}}
func.func @egraph(%a : index, %b : index) -> index {
  %a_eq = equivalence.class %a : index
  %one = arith.constant 1 : index
  %one_eq = equivalence.class %one : index
  %two = arith.constant 2 : index
  %two_eq = equivalence.class %two : index
  %a_shift_one = arith.shli %a_eq, %one_eq : index
  %a_times_two = arith.muli %a_eq, %two_eq : index
  %res_eq = equivalence.class %a_shift_one, %a_times_two : index
  func.return %res_eq : index
}

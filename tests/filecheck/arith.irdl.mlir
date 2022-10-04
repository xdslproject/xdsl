// RUN: xdsl-opt %s -t mlir | filecheck %s

// CHECK: "builtin.module"

"builtin.module"() ({
  "irdl.dialect"() ({
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.addf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.addi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"sum" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"carry" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.addui_carry"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.andi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.bitcast"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.ceildivsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.ceildivui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.cmpf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.cmpi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = []} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.constant"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.divf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.divsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.divui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.extf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.extsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.extui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.fptosi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.fptoui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.floordivsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.index_cast"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.maxf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.maxsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.maxui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.minf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.minsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.minui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.mulf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.muli"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"operand" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.negf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.ori"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.remf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.remsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.remui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.sitofp"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.shli"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.shrsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.shrui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.subf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.subi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.truncf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.trunci"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.uitofp"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.xori"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"condition" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"true_value" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"false_value" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "arith.select"} : () -> ()
  }) {name = "arith"} : () -> ()
}) : () -> ()


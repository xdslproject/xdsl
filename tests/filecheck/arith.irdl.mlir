// RUN: xdsl-opt %s -t mlir | filecheck %s

// CHECK: "builtin.module"

"builtin.module"() ({
  "irdl.dialect"() ({
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "addf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "addi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"sum" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"carry" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "addui_carry"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "andi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "bitcast"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "ceildivsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "ceildivui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "cmpf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "cmpi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = []} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "constant"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "divf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "divsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "divui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "extf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "extsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "extui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "fptosi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "fptoui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "floordivsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "index_cast"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "maxf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "maxsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "maxui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "minf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "minsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "minui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "mulf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "muli"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"operand" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "negf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "ori"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "remf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "remsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "remui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "sitofp"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "shli"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "shrsi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "shrui"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "subf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "subi"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "truncf"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "trunci"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"in" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"out" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "uitofp"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"lhs" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"rhs" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "xori"} : () -> ()
    "irdl.operation"() ({
      "irdl.operands"() {params = [#irdl.named_type_constraint<"condition" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"true_value" : #irdl.any_type_constraint>, #irdl.named_type_constraint<"false_value" : #irdl.any_type_constraint>]} : () -> ()
      "irdl.results"() {params = [#irdl.named_type_constraint<"result" : #irdl.any_type_constraint>]} : () -> ()
    }) {name = "select"} : () -> ()
  }) {name = "arith"} : () -> ()
}) : () -> ()


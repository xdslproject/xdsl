"builtin.module"() ({
  "irdl.dialect"() <{sym_name = "testd"}> ({
    "irdl.type"() <{sym_name = "singleton"}> ({
    ^bb0:
    }) : () -> ()
    "irdl.type"() <{sym_name = "parametrized"}> ({
      %1 = "irdl.any"() : () -> !irdl.attribute
      %2 = "irdl.is"() <{expected = i32}> : () -> !irdl.attribute
      %3 = "irdl.is"() <{expected = i64}> : () -> !irdl.attribute
      %4 = "irdl.any_of"(%2, %3) : (!irdl.attribute, !irdl.attribute) -> !irdl.attribute
      "irdl.parameters"(%1, %4) : (!irdl.attribute, !irdl.attribute) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "any"}> ({
      %0 = "irdl.any"() : () -> !irdl.attribute
      "irdl.results"(%0) <{variadicity = #irdl<variadicity_array[ single]>}> : (!irdl.attribute) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

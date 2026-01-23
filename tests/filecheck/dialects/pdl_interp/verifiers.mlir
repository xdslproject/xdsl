// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// CHECK: Region argument type !pdl.attribute does not match range element type !pdl.type

pdl_interp.func @argument_mismatch(%range: !pdl.range<type>) {
    pdl_interp.foreach %val : !pdl.attribute in %range {
        pdl_interp.continue
    } -> ^final
    ^final:
        pdl_interp.finalize
}

// -----

// CHECK: Region must have exactly one argument

"pdl_interp.func"() <{sym_name = "invalid_argcount", function_type = (!pdl.range<type>) -> ()}> ({
    ^bb0(%range : !pdl.range<type>):
    "pdl_interp.foreach"(%range) [^final] ({
        ^bb1(%val : !pdl.type, %extraval: !pdl.type):
            "pdl_interp.continue"() : () -> ()
    }) : (!pdl.range<type>) -> ()
    ^final:
    "pdl_interp.finalize"() : () -> ()
}) : () -> ()

// -----

// CHECK: Region must not be empty

"pdl_interp.func"() <{sym_name = "invalid_argcount", function_type = (!pdl.range<type>) -> ()}> ({
    ^bb0(%range : !pdl.range<type>):
    "pdl_interp.foreach"(%range) [^final] ({}) : (!pdl.range<type>) -> ()
    ^final:
    "pdl_interp.finalize"() : () -> ()
}) : () -> ()

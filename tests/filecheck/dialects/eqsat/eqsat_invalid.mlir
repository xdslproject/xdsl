// RUN: xdsl-opt --verify-diagnostics %s | filecheck %s


%val = "test.op"() : () -> index
%val_eq = eqsat.eclass %val : index
%val_eq_eq = eqsat.eclass %val_eq : index

// CHECK: Operation does not verify: A result of an eclass operation cannot be used as an operand of another eclass.

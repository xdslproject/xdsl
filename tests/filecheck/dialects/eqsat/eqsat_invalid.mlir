// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

%val = "test.op"() : () -> index
%val_eq = eqsat.eclass %val : index
%val_eq_eq = eqsat.eclass %val_eq : index

// CHECK: Operation does not verify: A result of an eclass operation cannot be used as an operand of another eclass.

// -----

%val_eq_eq = eqsat.eclass : index

// CHECK: Operation does not verify: Eclass operations must have at least one operand.

// -----

%val = "test.op"() : () -> index
%val_eq_eq = eqsat.eclass %val : index
"test.op"(%val) : (index) -> ()

// CHECK: Operation does not verify: Eclass operands must only be used by the eclass.

// -----

%val = "test.op"() : () -> index
%val_eq = eqsat.eclass %val, %val : index

// CHECK: Operation does not verify: Eclass operands must only be used once by the eclass.

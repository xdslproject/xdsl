// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

%val = "test.op"() : () -> index
%val_eq = equivalence.class %val : index
%val_eq_eq = equivalence.class %val_eq : index

// CHECK: Operation does not verify: A result of an e-class operation cannot be used as an operand of another e-class.

// -----

%val_eq_eq = equivalence.class : index

// CHECK: Operation does not verify: E-class operations must have at least one operand.

// -----

%val = "test.op"() : () -> index
%val_eq_eq = equivalence.class %val : index
"test.op"(%val) : (index) -> ()

// CHECK: Operation does not verify: E-class operands must only be used by the e-class.

// -----

%val = "test.op"() : () -> index
%val_eq = equivalence.class %val, %val : index

// CHECK: Operation does not verify: E-class operands must only be used once by the e-class.

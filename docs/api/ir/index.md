# IR Data Structures

## [Attribute](attribute.md)

Attributes represent compile-time information in xDSL.
All attributes are immutable, equatable, and hashable.

## [SSAValue](ssa_value.md)

Run-time information is represented as SSAValue.
SSA stands for [static single-assignment](https://en.wikipedia.org/wiki/Static_single-assignment_form), meaning that each value is only ever assigned once, and never reassigned.
This property is useful when performing transformations such as dead-code elimination, constant folding, and other compiler transformations.

## [IRNode](ir_node.md)

Code is represented as a recursive data structure, where [operations](operation.md) contain a doubly-linked list of [regions](region.md), which contain a doubly linked list of [blocks](block.md), which contain operations.

## [Operation](operation.md)

Operations are rich objects that carry some meaning in their IR.
They can be used to represent integer addition (`arith.addi`), a module of code (`builtin.module`), a function declaration or definition (`func.func`), and many more.

## [Block](block.md)

Blocks are a list of operations.
They have a (possibly empty) list of operands, that can be used to model function arguments for use in the function body, or loop arguments.

## [Region](region.md)

Regions are a list of blocks.

## [Dialect](dialect.md)

A dialect is a grouping of [attributes](attribute.md) and [operations](operation.md).

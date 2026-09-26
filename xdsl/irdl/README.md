# Design document for xDSL's constraint system

xDSL utilizes a constraint solving based system to power much of the functionality for defining and using operations. While powerful, this system can be difficult to understand, and is complicated by edge cases and attempts to maintain backwards compatibility and compatibility with upstream MLIR. The purpose of this document is to give an overview of the system.

## Introduction

In xDSL, the `Operation` class is simply a data structure providing access to the operands, results, regions, successors, properties, and attributes of the operation. The first four of these constructs are stored and accessed as a flat tuple, and the last two are unstructured dictionaries.

When operation types are defined in a dialect, they can add the following things to this operation class, of which only the first is mandatory:
- a name: used when printing the method (both in custom and generic formats),
- a verify method: which determines whether an operation is a valid member of this operation type,
- a build method: operation types typically define an `__init__` method to provide a more suitable interface for constructing them,
- accessors: functions (or python properties/descriptors) which allow more fine grained access to the data the operation contains, and offer ways to set this data,
- parsing and printing: methods for parsing and printing can be given to give the operation a custom IR format.

Take, for example, the relatively simple `arith.addi` operation. This operation takes two signless integers of the same type, and returns their addition as another signless integer of the same type. It also specifies a property for overflow flags. Let's observe what functionality is added by the `AddiOp` class:
- Name: the name "arith.addi" is given.
- Verification: The class verifies that there are precisely two operands, that they have the same type, and that this type is a signless integer (or a vector/tensor of signless integers). It further verifies that there is precisely one result, with the same type as the inputs, and verifies that there are no successors, and no regions. Lastly it verifies that there is always one property, with the name "overflowFlags".
- Build method: the default builder for `Operation` would require that tuples of operands and result types be specified, as well as a dictionary of properties. The builder for "arith.addi" instead takes precisely two operands, does not require the result type, and takes just the overflow attribute as an optional argument. The result type is then inferred from the operand types.
- Accessors: The class defines accessors for the "lhs" and "rhs" operands, the result, and overflow property. These have the appropriate python typing.
- Custom format: The operation defines a custom format, and methods for parsing and printing this custom format.

We note the operation also specifies traits, but they are not managed by the constraint system and so are not the focus of this document.

Overall, this is a lot of functionality to be implemented for every operation, and so xDSL introduces a layer `IRDLOperation` on top of `Operation` to simplify the definition of these methods. This layer is loosely based on the "irdl" dialect, an IR for defining operations. Each operation now defines a collection of named operands and results (much like in MLIR tablegen), each of which govern some slice of the operations operands/results. Each of these named operands/results is given a constraint, which instructs xDSL how to build the appropriate methods above. Properties of the operation can also be defined using constraints in this way. Lets see what the definition of `arith.addi` looks like (with flattened superclasses and no traits, omitted `__init__` method).

```python
@irdl_op_definition
class AddiOp(IRDLOperation):
    name = "arith.addi"

	T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

	overflow_flags = prop_def(
        IntegerOverflowAttr,
        default_value=IntegerOverflowAttr("none"),
        prop_name="overflowFlags",
    )

	assembly_format = (
        "$lhs `,` $rhs (`overflow` `` $overflowFlags^)? attr-dict `:` type($result)"
    )
```

Here, the operands of the operation have been split into a `lhs` operand and a `rhs` operand. Here the constraints enforce that each of the types of `lhs`, `rhs`, and `result` satisfy `signlessIntegerLike`, and the `VarConstraint` ensures that these types are all identical. We see that the `overflowFlags` property is enforced to be a `IntegerOverflowAttr`, and is given a default value, which `IRDLOperation` uses when defining accessors, printers, and parser. Finally, the operation defines an `assembly_format`, which is used to generate printing and parsing methods. This may seem disjoint from the constraint system, but the generated parsing methods rely on constraints for _inference_. Above, the parse method must assign types to `lhs` and `rhs`, but these are not given explicitly. From the constraints, it knows that these types must be the same as the type of `result`, which is given explicitly. In this way, constraints become central to much of the functionality provided by `IRDLOperation`.

## Scope

Constraints perform two roles, _verification_ and _inference_.
- Verification is the more obvious of the two roles. Given a fully formed operation, we can verify the operation by passing the appropriate segments of data to the appropriate constraints, calling their `verify` method.
- The need for inference is less clear but arguably more important than verification. A constraint can sometimes allow us to infer the value or type that should be present in an operation from context. This can sometimes occur when the constraint is specific enough that there is only one value that ever satisfies it, though sometimes it is necessary to know other information about the operation. With the `arith.addi` example above, the constraint `T = VarConstraint("T", signlessIntegerLike)` was able to infer the type of `lhs` during parsing, but only because the type of `result` is given.

Verification is mainly used to generate the `verify` method on operations, however it is also required by inference, as it used to populate constraint variables (see below TODO). Inference is crucial for parsing using the assembly format. Ideally, xDSL would also automatically define builders/`__init___` methods with constraints, but this is not currently implemented (partially due to a lack of satisfying api).

The constraint system comes with some "core" constraints, which can be combined to produce more complex behaviour, but it is also fully extendable; users can define their own constraints and use these within their operations.

## Types of constraint

There are three classes of constraints: attribute constraints subclassing `AttrConstraint`, range constraints subclassing `RangeConstraint`, and integer constraints subclassing `IntConstraint`.

TODO

## Verification and solving constraints

## Inference

## The any_of constraint

- Difficulties for inference
- Acts more like dispatch
- Bases of constraints

## Interfacing between python and constraints

- Python types from constraints
- Building constraints with python

## The declarative assembly format

TODO - intro, describe the data and phases

### Format directives

- Describe operations

### Parsing

### Printing

### Typeable directives

### The optional group

- Anchors
- Optionally parsable

### Custom directives

- Mainly talk about parameters, and get/set methods

### Corner cases

Some corner cases exist for the declarative assembly format to deal with:
- Segment size properties
- symbol names
- DenseArrayAttribute
- Typed attributes
- Attributes with known bases
- Unit attributes

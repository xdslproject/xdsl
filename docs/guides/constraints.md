# Guide to xDSL's constraint system

xDSL utilizes a constraint-solving-based system to power much of the functionality for
defining and using operations. While powerful, this system can be difficult to
understand, and is complicated by edge cases and attempts to maintain backwards
compatibility and compatibility with upstream MLIR. The purpose of this document is to
give an overview of the system.

## Introduction

In xDSL, the `Operation` class is simply a data structure providing access to the
operands, results, regions, successors, properties, and attributes of the operation. The
first four of these constructs are stored and accessed as a flat tuple, and the last two
are unstructured dictionaries.

When operation types are defined in a dialect, they can add the following things to this
operation class, of which only the first is mandatory:

- a name: used when printing the method (both in custom and generic formats),
- a verify method: which determines whether an operation is a valid member of this
  operation type,
- a build method: operation types typically define an `__init__` method to provide a
  more suitable interface for constructing them,
- accessors: functions (or Python properties/descriptors) which allow more fine-grained
  access to the data the operation contains, and offer ways to set this data,
- parsing and printing: methods for parsing and printing can be given to give the
  operation a custom IR format.

Take, for example, the relatively simple `arith.addi` operation. This operation takes
two signless integers of the same type, and returns their addition as another signless
integer of the same type. It also specifies a property for overflow flags. Let's observe
what functionality is added by the `AddiOp` class:

- Name: the name "arith.addi" is given.
- Verification: The class verifies that there are precisely two operands, that they have
  the same type, and that this type is a signless integer (or a vector/tensor of
  signless integers). It further verifies that there is precisely one result, with the
  same type as the inputs, and verifies that there are no successors or regions.
  Lastly it verifies that there is always one property, with the name "overflowFlags".
- Build method: the default builder for `Operation` would require that tuples of
  operands and result types be specified, as well as a dictionary of properties. The
  builder for "arith.addi" instead takes precisely two operands, does not require the
  result type, and takes just the overflow attribute as an optional argument. The result
  type is then inferred from the operand types.
- Accessors: The class defines accessors for the "lhs" and "rhs" operands, the result,
  and overflow property. These have the appropriate Python typing.
- Custom format: The operation defines a custom format, and methods for parsing and
  printing this custom format.

We note the operation also specifies traits, but they are not managed by the constraint
system and so are not the focus of this document.

Overall, this is a lot of functionality to be implemented for every operation, and so
xDSL introduces a layer `IRDLOperation` on top of `Operation` to simplify the definition
of these methods. This layer is loosely based on the "irdl" dialect, an IR for defining
operations. Each operation now defines a collection of named operands and results (much
like in MLIR tablegen), each of which govern some slice of the operations
operands/results. Each of these named operands/results is given a constraint, which
instructs xDSL how to build the appropriate methods above. Properties of the operation
can also be defined using constraints in this way. Let's see what the definition of
`arith.addi` looks like (with flattened superclasses and no traits, `__init__` method
omitted).

```python
@irdl_op_definition
class AddiOp(IRDLOperation):
    name = "arith.addi"

    # Simplified; actual arith.addi also allows vector/tensor of integers and IndexType
    T: ClassVar = VarConstraint.get("T", IntegerType)

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

Here, the operands of the operation have been split into a `lhs` operand and a `rhs`
operand. Here the constraints enforce that each of the types of `lhs`, `rhs`, and
`result` satisfy `IntegerType`, and the `VarConstraint` ensures that these types
are all identical. We see that the `overflowFlags` property is enforced to be a
`IntegerOverflowAttr`, and is given a default value, which `IRDLOperation` uses when
defining accessors, printers, and parsers. Finally, the operation defines an
`assembly_format`, which is used to generate printing and parsing methods. This may seem
disjoint from the constraint system, but the generated parsing methods rely on
constraints for _inference_. Above, the parse method must assign types to `lhs` and
`rhs`, but these are not given explicitly. From the constraints, it knows that these
types must be the same as the type of `result`, which is given explicitly. In this way,
constraints become central to much of the functionality provided by `IRDLOperation`.

## Scope

Constraints perform two roles, _verification_ and _inference_.

- Verification is the more obvious of the two roles. Given a fully formed operation, we
  can verify the operation by passing the appropriate segments of data to the
  appropriate constraints, calling their `verify` method.
- The need for inference is less clear but arguably more important than verification. A
  constraint can sometimes allow us to infer the value or type that should be present in
  an operation from context. This can sometimes occur when the constraint is specific
  enough that there is only one value that ever satisfies it, though sometimes it is
  necessary to know other information about the operation. With the `arith.addi` example
  above, the constraint `T = VarConstraint("T", IntegerType)` was able to infer
  the type of `lhs` during parsing, but only because the type of `result` is given.

Verification is mainly used to generate the `verify` method on operations, however it is
also required by inference, as it is used to populate constraint variables (see
[Inference](#inference) below). Inference is crucial for parsing using the assembly
format. Ideally, xDSL would also automatically define builders/`__init__` methods with
constraints, but this is not currently implemented (partially due to a lack of
satisfying API).

The constraint system comes with some "core" constraints, which can be combined to
produce more complex behaviour, but it is also fully extendable; users can define their
own constraints and use these within their operations.

## Types of constraint

There are three classes of constraints: attribute constraints subclassing
`AttrConstraint`, range constraints subclassing `RangeConstraint`, and integer
constraints subclassing `IntConstraint`.

Attribute constraints (`AttrConstraint`) are the most common, and are used to constrain
a single attribute value. They are typically used for the types of operands, results,
and properties. These include:

- `AnyAttr` - Matches any attribute (the most permissive constraint)
- `BaseAttr` - Matches attributes of a specific base type
- `EqAttrConstraint` - Matches exactly one specific attribute value
- `AttrSetConstraint` - Matches one of a set of specific attributes
- `VarConstraint` - Links multiple attributes as equal, with an inner constraint
- `TypeVarConstraint` - For generic ParametrizedAttribute type variables
- `ParamAttrConstraint` - Matches a parameterized type and constrains its parameters
- `AnyOf` - Matches if any of the given constraints match (union)
- `AllOf` - Matches if all constraints match (intersection)
- `MessageConstraint` - Wraps a constraint to provide a custom error message
- `SizedConstraint` - Constrains the length of a Sized attribute

`AnyOf` and `AllOf` constraints can also be formed using the `|` and `&` operators.

Range constraints (`RangeConstraint`) constrain a sequence of attributes, and are
primarily used for operands and results that may have variable arity (including optional
operands). These include:

- `AnyRange` - Matches any range of attributes (the most permissive constraint)
- `RangeOf` - Each element in the range must satisfy the given attribute constraint
- `SingleOf` - Range must contain exactly one element satisfying the constraint
- `RangeVarConstraint` - Links multiple ranges as equal, with an inner constraint
- `RangeLengthConstraint` - Constrains both the contents and length of a range

The length of a range can also be constrained by
`range_constraint.of_length(int_constraint)`.

Integer constraints (`IntConstraint`) constrain integer values, such as dimension sizes,
bit widths, or counts. These include:

- `AnyInt` - Matches any integer (the most permissive constraint)
- `EqIntConstraint` - Matches exactly one specific integer value
- `NotEqualIntConstraint` - Matches any integer except the given value
- `IntSetConstraint` - Matches one of a set of specific integers
- `AtLeast` - Matches integers >= min
- `AtMost` - Matches integers <= max
- `IntVarConstraint` - Links multiple integers as equal, with an inner constraint
- `IntTypeVarConstraint` - For generic integer type variables

Integer constraints are less commonly used directly, but are essential for operations
that manipulate shapes or sizes. They are also used internally to ensure the correct
arity of operands.

## Building constraints in Python

Constraints can be instantiated directly, but many provide smart `get` constructors that
accept shorthand inputs and automatically convert them to the appropriate constraint
type. The conversion is performed by `irdl_to_attr_constraint`, which for example
converts attribute instances to `EqAttrConstraint(attr)` and attribute classes
(`type[Attribute]`) to `BaseAttr(cls)`. TypeVars (from Python's typing module) are
converted to `TypeVarConstraint` using their bound, and Union types become `AnyOf`
constraints.

For example:

```python
# These are equivalent:
VarConstraint("T", BaseAttr(IntegerType))
VarConstraint.get("T", IntegerType)  # IntegerType is auto-converted to BaseAttr

# Union types become AnyOf:
IntegerType | FloatType  # → AnyOf(BaseAttr(IntegerType), BaseAttr(FloatType))

# Attribute instances become EqAttrConstraint:
i32  # → EqAttrConstraint(IntegerType(32, Signedness.SIGNLESS))
```

Generic attribute types are processed in two stages. First, the base class is converted
to a constraint (typically a `ParamAttrConstraint` with constraints for each parameter
position). Then, each type argument is recursively converted, and the resulting
constraints replace the `TypeVarConstraint` placeholders in the base constraint via
`mapping_type_vars`. This allows `MyAttr[IntegerType]` to become a constraint that
checks both the base type and that parameters match `IntegerType`.

All constraints must implement `mapping_type_vars`, which takes a mapping from
`TypeVar` to concrete constraints and returns a new constraint with those type
variables substituted. The implementation must call `mapping_type_vars` on all nested
subconstraints, propagating the substitution through the entire constraint tree.

This automatic conversion makes constraint definitions significantly more concise and
readable, especially for complex nested constraints. The same automatic conversion also
applies when declaring operation fields, e.g.

```python
# a and b are equivalent, as are c and d
a = operand_def()
b = operand_def(AnyAttr())

c = operand_def(BaseAttr(IntegerType))
d = operand_def(IntegerType)

# e and f are also equivalent, by converting the attribute constraint to a range constraint
e = var_operand_def(IntegerType)
f = var_operand_def(RangeOf(IntegerType))
```

The `get` methods also perform certain simplifications, for example
`AttrSetConstraint.get(attr)` becomes `EqAttrConstraint(attr)` or `ParamAttrConstraint`
with all `EqAttrConstraint` parameters can simplify to a single `EqAttrConstraint`.
Similar simplifications can occur during `mapping_type_vars` calls.

## Constraint generics and Python typing

Constraints are generic over Python types. For example, `BaseAttr[IntegerType]` is a
constraint that accepts `IntegerType` instances. This generic parameter flows through
the constraint system:

```python
class BaseAttr(AttrConstraint[AttributeCovT]):
    attr: type[AttributeCovT]
```

The generic type parameter is used to type the return value of `infer` and to type
accessors on operations. When an operand is defined as `operand_def(T)` where
`T = VarConstraint("T", BaseAttr(IntegerType))`, the accessor for that operand is
typed to return `IntegerType` rather than the generic `Attribute`. This provides
static type checking for operation builders and accessors.

## Verification and solving constraints

Verification is the process of checking that an attribute, range, or integer value
satisfies a constraint. Each constraint implements a `verify` method that takes the
value to check and a `VerificationContext`, and either returns silently (success) or
raises a `VerifyException` (failure).

### The verify method

The `verify` method is the core of constraint checking. Each constraint type has a
verify method with a similar signature:

```python
# Attribute constraints
@abstractmethod
def verify(
    self,
    attr: Attribute,
    constraint_context: VerificationContext,
) -> None:
    ...

# Range constraints
def verify(
    self,
    attrs: tuple[Attribute, ...],
    constraint_context: VerificationContext,
) -> None:
    ...

# Integer constraints
def verify(
    self,
    value: int,
    constraint_context: VerificationContext,
) -> None:
    ...
```

When verification fails, a `VerifyException` is raised with a descriptive error message.
For example, `BaseAttr(IntegerType)` will raise an exception if given a `FloatType`, and
`EqAttrConstraint(attr)` will fail if the attribute doesn't match exactly.

### Composing constraints

Constraints often contain other constraints and delegate verification to them. For
example, `ParamAttrConstraint` verifies that an attribute is of a specific parameterized
type, and then verifies each of its parameters against the corresponding inner
constraints:

```python
@dataclass(frozen=True)
class ParamAttrConstraint(AttrConstraint):
    base: type[ParametrizedAttribute]
    parameters: tuple[AttrConstraint, ...]

    def verify(
        self,
        attr: Attribute,
        constraint_context: VerificationContext,
    ) -> None:
        # Verify the base type
        if not isinstance(attr, self.base):
            raise VerifyException(...)
        # Verify each parameter by calling its constraint's verify method
        for i, param_constraint in enumerate(self.parameters):
            param_constraint.verify(attr.parameters[i], constraint_context)
```

The critical point is that the same `VerificationContext` must be passed through to all
nested constraint verification calls. This ensures that variable assignments are shared
across the entire constraint tree.

Further, it is essential that a `VerifyException` in a nested constraint causes a
`VerifyException` in the parent constraint, and so a `VerifyException` should only be
caught if another one is immediately thrown in the catch block.

### VarConstraint and VerificationContext

`VarConstraint` is the primary reason constraints need a verification context. A
`VarConstraint` ensures that all the attributes its verify method is called on are
equal. When a `VarConstraint` verifies an attribute, it calls
`context.set_attr_variable(self.name, attr)` to record the assignment. This method
returns `True` if the variable was newly set, and `False` if it was already set to the
same value. `VarConstraint` uses this to determine if its contained constraint should
be run.

The `VerificationContext` maintains a mapping from variable names to their assigned
values (attributes, ranges, or integers). The key invariant is that once a variable is
assigned, it cannot be reassigned to a different value. If `set_attr_variable` is called
twice with the same name but different attributes, it raises a `VerifyException`:

```text
VerifyException: attribute i32 expected from variable 'T', but got i64
```

This is the mechanism for actually ensuring attribute equality. The context can store
attributes, ranges, and integers which are used respectively by `VarConstraint`,
`RangeVarConstraint`, and `IntVarConstraint`.

Constraints also implement a `variables` method that returns the set of variable names
they promise to set during verification. This allows the constraint system to know which
variables will be available for inference after verification completes. For composite
constraints like `AllOf` or `ParamAttrConstraint`, the `variables` method aggregates the
variables from all nested constraints.

```python
VarConstraint.get("T", IntegerType).variables()  # Returns {"T"}
AllOf(VarConstraint.get("A"), VarConstraint.get("B")).variables()  # Returns {"A", "B"}
```

### Limitations: One-pass verification

The verification context has an important limitation: verification is single-pass.
Variables are populated as constraints are verified in order, and a constraint cannot
query a variable's value before it has been set.

Constraint solving should not depend on the order that verification occurs in, as this
ordering is hidden from the user and cannot be controlled, and would likely make such
constraints hard to maintain and prone to subtle bugs. Therefore constraints should
never depend on getting the values of other variables, including calling the inference
methods of other constraints.

For example, a constraint should not try to infer the value of variable `"A"` during
verification of variable `"B"`, as this assumes `"A"` has already been set. Such code
will fail unpredictably depending on the order operands/results are defined.

## Inference

Inference is the process of determining an attribute, range, or integer value from a
constraint when that value is not explicitly provided. This is essential during parsing,
where types may be omitted and must be reconstructed from context.

Inference relies on constraint variables that have been populated during verification.
When parsing an operation, verification runs first to collect variable assignments into
the context. The inference phase then queries this context to fill in missing values.

For example, if operand `x` is constrained by `VarConstraint("T")`, then after
verification the context knows that `"T" = i32`. When inferring the type of `x`, the
constraint can query the context for `T` and return `i32`.

The solved variables are passed in through an `InferenceContext`, which provides
read-only access to these assignments:

- `get_variable(name)` - Get an attribute variable
- `get_range_variable(name)` - Get a range variable
- `get_int_variable(name)` - Get an integer variable

Unlike `VerificationContext`, the inference context does not allow new assignments.

Not all constraints can infer values in all situations. The `can_infer` method
determines whether a constraint has enough information to perform inference, taking the
set of known variable names as a parameter. For example, `EqAttrConstraint(attr)` can
always infer (it knows the exact value), `VarConstraint("T", constraint)` can infer if
`"T"` is known or if its inner constraint can infer, and `BaseAttr(IntegerType)` cannot
infer (there are many possible integer types). If `can_infer` returns `True`, the
`infer` method must succeed. Since the constraint system knows which
variables will be set by verification (via the `variables` method), it can ensure that
inference will succeed when building operations.

## The `AnyOf` constraint

A true `AnyOf` constraint that accepts any matching subconstraint would complicate both
verification and inference. During verification, the system wouldn't know which
variables would be set (different subconstraints might set different `VarConstraint`
names), requiring backtracking. During inference, it wouldn't know which subconstraint
to call `infer` on.

Instead, `AnyOf` acts as a dispatch mechanism. It uses the base class of the attribute
to deterministically select which subconstraint to use. This requires knowing which base
classes each constraint accepts, which is provided by the `get_bases` method:

```python
def get_bases(self) -> set[type[Attribute]] | None:
    """
    Get a set of base types that can satisfy this constraint, if there exists
    a finite collection, or None otherwise.
    """
```

Constraints that want to be usable in `AnyOf` must implement `get_bases`. For example,
`BaseAttr(IntegerType)` or `EqAttrConstraint(i32)` returns `{IntegerType}`, while `AnyOf`
aggregates the bases from all its subconstraints. Constraints with `get_bases() == None`
cannot participate in dispatch (there's no base class to dispatch on).

The bases returned must be final classes: those decorated with `@irdl_attr_definition`.
`AnyOf` allows at most one constraint with `get_bases() == None` (typically
`BaseAttr(SomeAbstractBase)`). This abstract constraint acts as a fallback for
attributes that don't match any of the concrete base classes, and is checked at runtime
after dispatch to the concrete constraints fails.

A common pattern is wanting to write `i32 | i64` to mean "either i32 or i64". This
doesn't work with `AnyOf` since both have the same base class (`IntegerType`). Instead,
such cases are converted to `AttrSetConstraint({i32, i64})`, which explicitly lists the
allowed values.

### Custom constraints

Users can define custom constraints by subclassing `AttrConstraint`, `RangeConstraint`,
or `IntConstraint` and implementing the required methods:

```python
@dataclass(frozen=True)
class EvenIntConstraint(IntConstraint):
    """Constraint that accepts only even integers."""

    def verify(self, i: int, context: VerificationContext) -> None:
        if i % 2:
            raise VerifyException(f"Expected even integer, got {i}")

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> IntConstraint:
        # No type variables in this constraint, return self
        return self
```

Custom constraints must implement `verify` and `mapping_type_vars`. For inference
support, also implement `can_infer` and `infer`. Attribute should implement `get_bases`
correctly to ensure `AnyOf` compatibility.

## Example: Parsing an arith.addi operation

To see how constraints work together, consider parsing this IR:

```mlir
%result = arith.addi %lhs, %rhs : i32
```

The assembly format specifies that `type($result)` appears explicitly, but operand types
are omitted. When `AddiOp` is defined, the constraint system collects all variables that
will be set (via the `variables` method). It knows that `"T"` will be set by the result
constraint. It can then check that the operand types are inferrable from `"T"` using the
`can_infer` method. This allows the system to statically verify that inference will
succeed: once the result type is known, the operands can be inferred.

The constraint system handles this as follows:

1. Parsing operands: The parser encounters `%lhs` and `%rhs` first, but their types
   are not yet known. It creates SSA values with placeholder types.

2. Verification phase: When parsing `: i32` for the result, the parser calls `verify`
   on the result's constraint. This sets `T = i32` in the context via `VarConstraint`.

3. Inference phase: Now that `"T"` is known, calling `infer` fills in `i32` as the type
   for both `%lhs` and `%rhs`.

4. Default values: The `overflowFlags` property is not specified in the IR. Since
   it has a `default_value` in its `prop_def`, this default is used. Properties cannot
   be inferred from constraints—they must either be explicitly provided or use defaults.

This flow allows the IR to be concise while the constraint system reconstructs all
necessary type information.

## See also

- `xdsl/irdl/constraints.py` - Full implementation of the constraint system
- `xdsl/dialects/arith.py` - Example dialect using constraints in practice
- `docs/guides/assembly_format.md` - Declarative assembly format (forthcoming)

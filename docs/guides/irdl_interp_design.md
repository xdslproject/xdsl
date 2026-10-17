# IRDL Interp Dialect Design

## Motivation

The goal of `irdl_interp` is to provide a runtime for constraint solving in IR, allowing constraint verification, inference, and accessor generation to be expressed as optimizable IR rather than Python code. This mirrors how `pdl_interp` provides primitives for pattern matching.

Instead of using `VerificationContext`/`InferenceContext` (Python implementation artifacts), constraint variables are represented as **SSA values** in the IR, allowing constraint solving to be expressed as dataflow.

## Core Design Principles

1. **Variables as SSA values**: Constraint variables are first-class SSA values, not entries in a dictionary
2. **Expressions over variables**: Constraint variables are leaf expressions; compound expressions (addition, scalar multiplication) enable algebraic reasoning
3. **Unify as constraint solving**: `unify` solves equations of the form `expr = c` where `c` is a known constant
4. **Bounds verification**: Inequalities (`>=`, `<=`) are verified on expressions, returning `i1` for explicit assertion
5. **Minimal surface area**: Start with the smallest set of operations needed

## Types

### Constraint Expressions

Constraint expressions represent symbolic computations that can be solved or verified:

```
!irdl_interp.expr.attr   - attribute constraint expression (variable or concrete attribute)
!irdl_interp.expr.range  - range constraint expression (variable or concrete range)
!irdl_interp.expr.int    - integer constraint expression (variable, constant, or compound expression)
```

**Design rationale**: All constraint types use the `.expr` suffix to indicate they support symbolic reasoning. Currently, only integer expressions support compound operations (addition, scalar multiplication), but the naming is consistent across all types for future extensibility.

### Concrete Values

```
!irdl_interp.attr            - wrapper for a concrete Attribute value
tensor<?x!irdl_interp.attr>  - concrete range (dynamic-size tensor of attribute values)
index                        - concrete integer (MLIR index type)
```

**Rationale for index**: Constraint integers are used for sizes, counts, and indices. MLIR's `index` type is the natural fit and works directly with `tensor.dim` and other tensor operations.

**Rationale for tensor<?x!irdl_interp.attr>**: Ranges are sequences of attributes with potentially dynamic sizes. Using MLIR's tensor type allows runtime-determined lengths.

### Structure Handles

```
!irdl_interp.operation       - wrapper for an Operation
!irdl_interp.region          - wrapper for a Region
!irdl_interp.block           - wrapper for a Block (successor)
```

**Rationale**: Operations, regions, and blocks are first-class handles that can be passed around and queried for their properties (counts, contents, etc.).

## Operations

### Operation Access

To work with operations in constraint solving, we need to access their structure:

```
!irdl_interp.operation       - wrapper for an Operation
```

```mlir
# Extract property by name (returns attribute)
%prop = irdl_interp.get_prop %op, "operandSegmentSizes"

# Get property count (for verifying no extra properties)
%count = irdl_interp.get_prop_count %op

# Extract all operand types (gives length + individual access via tensor.extract)
%operand_types = irdl_interp.get_operand_types %op

# Extract all result types
%result_types = irdl_interp.get_result_types %op

# Extract all regions (gives length + individual access via tensor.extract)
%regions = irdl_interp.get_regions %op

# Extract all successors (gives length + individual access via tensor.extract)
%successors = irdl_interp.get_successors %op

# Extract all parameters from a parametrized attribute
%params = irdl_interp.get_params %attr

# Check attribute base type (returns i1)
%is_int = irdl_interp.check_base %attr, "IntegerType"

# Assert condition with message
irdl_interp.assert %is_int, "Expected IntegerType"
```

**Assembly format**: All types elided (results are always the same types).

**Design decisions**:
- Return whole collections rather than indexed access
- Length available via tensor operations (`tensor.dim`)
- Individual elements accessible via `tensor.extract`
- Regions and blocks are opaque handles (contents accessed separately)

**Design decisions**:
- Return whole collections rather than indexed access
- Length available via tensor operations
- Individual elements accessible via `tensor.extract`

### Variable Creation

Create an uninitialized constraint expression variable:

```mlir
%x = irdl_interp.new_var.attr : !irdl_interp.expr.attr
%y = irdl_interp.new_var.range : !irdl_interp.expr.range
%z = irdl_interp.new_var.int : !irdl_interp.expr.int
```

**Assembly format**: Type is elided (encoded in operation name).

Variables are leaf expressions and can be used in compound expressions (for integers) or unified with concrete values.

### Constant Creation

Create concrete attribute or integer values lifted to expressions:

```mlir
%attr = irdl_interp.constant.attr i32  // Attribute expression
%c = arith.constant 2 : index
%expr_c = irdl_interp.constant.int %c : !irdl_interp.expr.int  // Integer expression
```

**Assembly format**: Attribute/constant is passed as an attribute operand.

For integers, use `arith.constant` to create `index` values, then lift to expressions:

```mlir
%two = arith.constant 2 : index
%expr_two = irdl_interp.constant.int %two : !irdl_interp.expr.int
irdl_interp.unify.int %operand_count, %expr_two : !irdl_interp.expr.int, index
```

For ranges, use standard tensor operations to construct `tensor<?x!irdl_interp.attr>` values.

### Integer Expression Operations

Integer expressions enable algebraic reasoning about operand/result counts, supporting:
- **Linear equalities**: Solve `expr = c` where `c` is known at runtime
- **Inequality verification**: Check `expr >= c` or `expr <= c`
- **Compound expressions**: Addition and scalar multiplication

```mlir
// Create integer variable
%N = irdl_interp.new_var.int : !irdl_interp.expr.int

// Lift concrete value to expression
%c = arith.constant 2 : index
%expr_c = irdl_interp.constant.int %c : !irdl_interp.expr.int

// Build compound expressions
%sum = irdl_interp.add.int %N, %expr_c : !irdl_interp.expr.int  // N + 2
%two = arith.constant 2 : index
%scaled = irdl_interp.scalar_mul.int %two, %N : !irdl_interp.expr.int  // 2*N

// Solve equation: expr = c
%result = irdl_interp.unify.int %expr, %c : !irdl_interp.expr.int, index -> i1
// Returns true if solvable, false if no integer solution

// Verify inequalities
%ge = irdl_interp.ge.int %expr, %c : !irdl_interp.expr.int, index -> i1  // expr >= c
%le = irdl_interp.le.int %expr, %c : !irdl_interp.expr.int, index -> i1  // expr <= c

// Assert verification result
irdl_interp.assert %ge, "Expected expr >= c"
```

**Design decisions**:
- All operations return `i1` for explicit error handling via `irdl_interp.assert`
- `unify.int` solves `expr = c`, not general `expr1 = expr2` (keeps solving tractable)
- Scalar multiplication enables SameVariadicSize constraints: `n * var = total`
- Inequalities verified on expressions, not just concrete values
- Compound expressions enable canonicalization: `x + x + 5 = total` → `2*x = total - 5`

**Solver semantics**:
- Variables start unbound
- `unify.int` attempts to solve the equation
- Returns `true` if integer solution exists, `false` otherwise
- Underdetermined systems: solver returns `true`, but `read.int` (future) will fail if value not uniquely determined

### Unify

The core constraint solving operation. Takes an expression and a concrete value, attempts to solve `expr = c`:

```mlir
// Integer unification (solves equation)
%result = irdl_interp.unify.int %expr, %c : !irdl_interp.expr.int, index -> i1

// Attribute unification (binding)
irdl_interp.unify.attr %attr : !irdl_interp.attr and %var : !irdl_interp.expr.attr

// Range unification (binding)
irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %var_range : !irdl_interp.expr.range
```

**Assembly format**: Each operand's type is printed inline after the operand. This is required because expression types cannot be inferred from the operation context.

**Semantics**:
- **Integer unify**: Solves `expr = c` for integer solutions
  - Returns `true` if solvable, `false` if no integer solution
  - For `expr = var` (variable), binds the variable
  - For `expr = compound`, solves linear equation
  - Example: `2*N = 5` returns `false` (no integer solution)
- **Attribute/Range unify**: Binds variable to concrete value
  - First unify with a value: sets the variable
  - Subsequent unify with same value: succeeds (idempotent)
  - Subsequent unify with different value: returns `false`
- Variable-variable unify is not supported; reuse the same SSA variable instead

**Return value semantics**:
- Returns `true` if the constraint is **satisfiable** (solution exists or binding succeeds)
- Returns `false` if there is an **explicit contradiction** (no solution possible)
- Does **not** fail if variables remain undersolved - that's checked later by `read.int` (Phase 2)
- **Non-failing**: The operation itself never raises an exception; use `irdl_interp.assert` to enforce constraints

**Examples**:
```mlir
// Successful binding
%N = irdl_interp.new_var.int
%five = arith.constant 5 : index
%expr_five = irdl_interp.constant.int %five
%solved = irdl_interp.unify.int %N, %expr_five : i1  // Returns true, N = 5

// No integer solution
%N = irdl_interp.new_var.int
%two = arith.constant 2 : index
%expr_two = irdl_interp.constant.int %two
%doubled = irdl_interp.scalar_mul.int %expr_two, %N
%five = arith.constant 5 : index
%expr_five = irdl_interp.constant.int %five
%solved = irdl_interp.unify.int %doubled, %expr_five : i1  // Returns false (2*N = 5 has no integer solution)

// Undersolved but satisfiable
%N = irdl_interp.new_var.int
%M = irdl_interp.new_var.int
%sum = irdl_interp.add.int %N, %M
%five = arith.constant 5 : index
%expr_five = irdl_interp.constant.int %five
%solved = irdl_interp.unify.int %sum, %expr_five : i1  // Returns true, but N and M undersolved
// Many solutions exist: (N=0,M=5), (N=1,M=4), etc.
// This is OK - additional constraints will narrow it down, or read.int will fail
```

### Integer Inequality Verification

Verify inequality constraints on integer expressions. These operations can **solve for variable bounds** when possible:

```mlir
// Greater-or-equal: expr >= c
%ge = irdl_interp.ge.int %expr, %c : !irdl_interp.expr.int, index -> i1

// Less-or-equal: expr <= c
%le = irdl_interp.le.int %expr, %c : !irdl_interp.expr.int, index -> i1
```

**Usage with assertion**:
```mlir
%count = irdl_interp.new_var.int
%two = arith.constant 2 : index
%expr_two = irdl_interp.constant.int %two
%ge = irdl_interp.ge.int %count, %expr_two : i1
irdl_interp.assert %ge, "Expected count >= 2"
```

**Solver semantics for inequalities**:
- Inequalities can **solve for variable bounds** when the expression is a single variable
- Example: `ge.int %N, 2` where `%N` is unbound → solver infers `N >= 2`
- Combined with `unify.int`: `unify.int %N, 5` and `ge.int %N, 2` → `N = 5` (satisfies both)
- Compound expressions: `ge.int (%N + 3), 5` → solver infers `N >= 2`

**Return value semantics**:
- Returns `true` if the constraint is **satisfiable** (no conflict detected)
- Returns `false` if there is an **explicit contradiction**
- Does **not** fail if variables remain undersolved - that's checked later by `read.int` (Phase 2)

**Examples**:
```mlir
// Satisfiable inequality
%N = irdl_interp.new_var.int
%two = arith.constant 2 : index
%expr_two = irdl_interp.constant.int %two
%ge = irdl_interp.ge.int %N, %expr_two : i1  // Returns true (N >= 2 is satisfiable)

// Contradiction
%N = irdl_interp.new_var.int
%five = arith.constant 5 : index
%expr_five = irdl_interp.constant.int %five
irdl_interp.unify.int %N, %expr_five  // Binds N = 5
%three = arith.constant 3 : index
%expr_three = irdl_interp.constant.int %three
%le = irdl_interp.le.int %N, %expr_three : i1  // Returns false (5 <= 3 is false)

// Undersolved but satisfiable
%N = irdl_interp.new_var.int
%two = arith.constant 2 : index
%expr_two = irdl_interp.constant.int %two
%ge = irdl_interp.ge.int %N, %expr_two : i1  // Returns true, N is still undersolved
// This is OK - N will be solved by later constraints or fail at read.int
```

**Design decisions**:
- Returns `i1` for explicit error handling via `irdl_interp.assert`
- Works on expressions, not just concrete values
- Enables `AtLeast`, `AtMost`, and `Optional` constraints
- **Participates in solving** - can infer variable bounds
- **Non-failing**: Returns `false` on conflict, doesn't raise exception

### Check Base

Verify an attribute has a specific base type:

```mlir
irdl_interp.check_base %attr : !irdl_interp.attr, "IntegerType"
irdl_interp.check_base %var : !irdl_interp.expr.attr, "IntegerType"
```

**Assembly format**: Operand type is printed (distinguishes concrete vs variable). Base type name as a bare string attribute. `attr-dict` at end for consistency.

**Semantics**:
- If the operand is a concrete attribute: verifies the base type, throws on mismatch
- If the operand is an unbound variable: no-op (constraint will be checked when variable is bound)
- If the operand is a bound variable: verifies the bound value's base type

## Design Philosophy

Constraint expressions should be thought of as **symbolic computations with a solver**:
- Expressions start as symbolic (variables or compounds)
- `unify` solves equations of the form `expr = c`
- `ge`/`le` verify inequalities and infer bounds
- Once solved, variables are bound to concrete values
- Using an undersolved variable fails at `read.int` (Phase 2)
- Operations return `i1` on conflict (non-failing), use `irdl_interp.assert` to enforce

## Open Questions

1. **Additional accessors**: 
   - Successors? Implemented via `irdl_interp.get_successors` (opaque handles)
   - Regions? Implemented via `irdl_interp.get_regions` (opaque handles)
   - Attribute parameters (for ParamAttrConstraint)? Implemented via `irdl_interp.get_params`
   - Operation attributes (vs properties)? No, these are being deprecated

2. **Attribute operations**: 
   - Extract parameters from parametrized attributes? Implemented via `irdl_interp.get_params`
   - Check attribute base type? Implemented via `irdl_interp.check_base`
   - Construct attributes from parameters? Not needed.

3. **Region and block operations**: Future work needed:
   - Get region body (iterate over operations in a region)
   - Check region properties (single-block, etc.)
   - Get block arguments and terminators
   - Iterate over successor blocks
   - Verify block signatures

4. **Integer expression extraction**: How to extract solved values for tensor operations?
   - Need `irdl_interp.read.int` operation (deferred to Phase 2)
   - Alternative: interpreter handles extraction implicitly

5. **Canonicalization**: What algebraic simplifications should be supported?
   - Constant folding: `x + 2 + 3` → `x + 5`
   - Coefficient combination: `2*x + 3*x` → `5*x`
   - Expression normalization: `x + x` → `2*x`

6. **Undersolved variable handling**: When should undersolved variables be detected as errors?
   - Current design: `read.int` fails if variable not uniquely determined
   - Alternative: Detect during canonicalization (eager error detection)
   - Trade-off: Early errors vs. more optimization opportunities

## Example: arith.addi verify function

The verify method for `arith.addi` (which has 2 operands and 1 result, all with the same type) would look like:

```mlir
func.func @verify_addi(%op: !irdl_interp.operation) {
  %operand_types = irdl_interp.get_operand_types %op
  %result_types = irdl_interp.get_result_types %op

  %c0 = arith.constant 0 : index
  %operand_count = tensor.dim %operand_types, %c0 : tensor<?x!irdl_interp.attr>
  %result_count = tensor.dim %result_types, %c0 : tensor<?x!irdl_interp.attr>

  %two = arith.constant 2 : index
  %one = arith.constant 1 : index
  irdl_interp.unify.int %operand_count : index and %two : index
  irdl_interp.unify.int %result_count : index and %one : index

  // Create constraint variable for the shared type
  %T = irdl_interp.new_var.attr

  // Extract individual operand/result types
  %lhs_type = tensor.extract %operand_types[0] : tensor<?x!irdl_interp.attr>
  %rhs_type = tensor.extract %operand_types[1] : tensor<?x!irdl_interp.attr>
  %result_type = tensor.extract %result_types[0] : tensor<?x!irdl_interp.attr>

  // Check each type satisfies VarConstraint("T", BaseAttr(IntegerType))
  // "Check every time" semantics - each use checks the constraint.
  // Optimization can eliminate redundant checks later.
  %lhs_check = irdl_interp.check_base %lhs_type : !irdl_interp.attr, "IntegerType"
  irdl_interp.assert %lhs_check, "Expected IntegerType for lhs"
  %rhs_check = irdl_interp.check_base %rhs_type : !irdl_interp.attr, "IntegerType"
  irdl_interp.assert %rhs_check, "Expected IntegerType for rhs"
  %result_check = irdl_interp.check_base %result_type : !irdl_interp.attr, "IntegerType"
  irdl_interp.assert %result_check, "Expected IntegerType for result"

  // Unify all types to ensure they are equal (VarConstraint "T" semantics)
  irdl_interp.unify.attr %lhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  irdl_interp.unify.attr %rhs_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
  irdl_interp.unify.attr %result_type : !irdl_interp.attr and %T : !irdl_interp.variable.attr

  // Check overflowFlags property
  %overflow_flags = irdl_interp.get_prop %op, "overflowFlags"
  %overflow_check = irdl_interp.check_base %overflow_flags : !irdl_interp.attr, "IntegerOverflowAttr"
  irdl_interp.assert %overflow_check, "Expected IntegerOverflowAttr"

  // Verify no extra properties (exactly 1 property expected)
  %prop_count = irdl_interp.get_prop_count %op
  irdl_interp.unify.int %prop_count : index and %one : index

  // Verify no regions (arith.addi has no regions)
  %regions = irdl_interp.get_regions %op
  %region_count = tensor.dim %regions, %c0 : tensor<?x!irdl_interp.region>
  %zero = arith.constant 0 : index
  irdl_interp.unify.int %region_count : index and %zero : index

  // Verify no successors (arith.addi has no successors)
  %successors = irdl_interp.get_successors %op
  %succ_count = tensor.dim %successors, %c0 : tensor<?x!irdl_interp.block>
  irdl_interp.unify.int %succ_count : index and %zero : index

  func.return
}
```

The generated code demonstrates:
- Operand/result counts match expected values
- **Check every time**: Each operand/result type is checked against the constraint individually
- **Unify for equality**: All types are unified to ensure they are equal (VarConstraint "T" semantics)
- Property count matches expected (combined with explicit property checks, ensures no extra properties)
- Region/successor counts match expected (ensures no unexpected regions or successors)

**VarConstraint lowering**: A `VarConstraint("T", constraint)` is lowered to:
1. Check the constraint on each concrete value that uses the variable (returns `i1`)
2. Assert the check passes (with error message)
3. Unify all values with the variable to ensure equality

This generates redundant checks, but optimization passes (CSE, canonicalization) can eliminate them. The redundancy makes the IR simpler to generate and easier to optimize.

**`check_base`** verifies the base type of an attribute and returns `i1` (1 if matches, 0 otherwise). This allows conditional logic and better AnyOf lowering.

**`assert`** takes an `i1` condition and a message string. Fails verification if the condition is false.

---

## OpDef to Verify Function Lowering

### Overview

An `OpDef` (IRDL operation definition) can be lowered to an irdl_interp verify function. This allows constraint verification to be expressed as optimizable MLIR rather than Python code.

### Scope (Phase 1)

**In scope:**
- Operand/result/region/successor count verification
- Operand and result type constraint verification
- Property verification (existence, constraints, no extra properties)
- Basic constraints: `BaseAttr`, `AnyAttr`, `VarConstraint`, `AnyOf`, `EqAttrConstraint`, `ParamAttrConstraint`, `RangeOf`, `RangeLengthConstraint`

**Out of scope (future phases):**
- Region shape verification (single-block, etc.)
- Successor verification (signatures, terminators)
- Trait verification (HasParent, HasCanonicalizer, etc.)
- Attribute verification (deprecated path)
- Optional properties

### Verification Phases

The generated verify function follows these phases in order:

#### Phase 1: Verify Structural Counts

Extract all structural components and verify counts match expected values.

```mlir
// Extract all components
%operand_types = irdl_interp.get_operand_types %op
%result_types = irdl_interp.get_result_types %op
%regions = irdl_interp.get_regions %op
%successors = irdl_interp.get_successors %op
%prop_count = irdl_interp.get_prop_count %op

%c0 = arith.constant 0 : index

// Verify operand count
%operand_count = tensor.dim %operand_types, %c0
%expected_operands = arith.constant N : index
irdl_interp.unify.int %operand_count : index and %expected_operands : index

// Verify result count
%result_count = tensor.dim %result_types, %c0
%expected_results = arith.constant M : index
irdl_interp.unify.int %result_count : index and %expected_results : index

// Verify region count
%region_count = tensor.dim %regions, %c0
%expected_regions = arith.constant R : index
irdl_interp.unify.int %region_count : index and %expected_regions : index

// Verify successor count
%succ_count = tensor.dim %successors, %c0
%expected_successors = arith.constant S : index
irdl_interp.unify.int %succ_count : index and %expected_successors : index
```

**Rationale:** Verifying counts first ensures the operation has the correct structure before checking types. All counts are verified in one phase for clarity.

#### Phase 2: Setup Constraint Variables

Collect all `VarConstraint` variable names from the operation's constraints, then create them upfront. Variables are stored in an `EmitContext` which is passed to all constraint lowering operations.

```mlir
%T = irdl_interp.new_var.attr  // For type variables
%N = irdl_interp.new_var.int   // For integer variables
```

**Rationale:** Creating variables upfront ensures they are available for both operand and result verification, enabling shared type variables across operands and results. The immutable `EmitContext` ensures variables are not accidentally mutated during constraint lowering.

#### Phase 3: Verify Operand Type Constraints

Extract each operand type and verify it satisfies its constraint.

```mlir
%type = tensor.extract %operand_types[%c0] : tensor<?x!irdl_interp.attr>

// BaseAttr example
%check = irdl_interp.check_base %type, "IntegerType" : i1
irdl_interp.assert %check, "Expected IntegerType"

// VarConstraint example (binds the variable)
irdl_interp.unify.attr %type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
```

**"Check every time" semantics:** Each operand type is checked against its constraint individually, even if multiple operands share the same `VarConstraint`. This generates redundant checks, but optimization (CSE) can eliminate them.

#### Phase 4: Verify Result Type Constraints

Same pattern as Phase 3, but for results. Shared type variables (from Phase 2) are unified with result types, ensuring all operands and results with the same `VarConstraint` have equal types.

```mlir
%type = tensor.extract %result_types[%c0] : tensor<?x!irdl_interp.attr>
%check = irdl_interp.check_base %type, "IntegerType" : i1
irdl_interp.assert %check, "Expected IntegerType"
irdl_interp.unify.attr %type : !irdl_interp.attr and %T : !irdl_interp.variable.attr
```

#### Phase 5: Verify Properties

Verify each property exists and satisfies its constraint, then verify no extra properties exist.

```mlir
// Check required property
%prop = irdl_interp.get_prop %op, "overflowFlags"
%check = irdl_interp.check_base %prop, "IntegerOverflowAttr" : i1
irdl_interp.assert %check, "Expected IntegerOverflowAttr"

// Check no extra properties
%expected_prop_count = arith.constant N : index
irdl_interp.unify.int %prop_count : index and %expected_prop_count : index
```

### Constraint Lowering Infrastructure

Constraints are lowered via the `emit_irdl_interp` method, which takes:
- `value: SSAValue` - The value to verify
- `builder: Builder` - For inserting operations
- `context: EmitContext` - Immutable context with all constraint variables
- `message: str | None = None` - Optional message prefix to prepend to default assert messages

**`EmitContext`** is created in Phase 2 (Setup Constraint Variables) and provides:
```python
@dataclass(frozen=True)
class EmitContext:
    _attr_variables: dict[str, SSAValue]
    _range_variables: dict[str, SSAValue]
    _int_variables: dict[str, SSAValue]

    @property
    def attr_variables(self) -> Mapping[str, SSAValue]: ...
    @property
    def range_variables(self) -> Mapping[str, SSAValue]: ...
    @property
    def int_variables(self) -> Mapping[str, SSAValue]: ...
```

Variables are accessed via dict syntax: `context.attr_variables["T"]`

**Message parameter**: When provided, the message is prepended to default assert messages with simple concatenation:
```
{message}{default_message}
```

This allows outer constraints (like `MessageConstraint`) to add context to error messages. The message propagates through nested constraints, creating a chain of context. Messages typically end with `: ` or similar separator for readability (e.g., `"Custom error: Expected integer"`).

### Unify Operations

All `irdl_interp.unify.*` operations return `i1` (boolean) indicating success or failure:
- **Returns `true`**: Unification succeeded, variables are bound
- **Returns `false`**: Unification failed (conflict), no error raised

This allows composing unification results with boolean logic (e.g., for `AnyOf`, `AttrSetConstraint`).
The caller is responsible for asserting the result:

```mlir
%result = irdl_interp.unify.attr %value : !irdl_interp.attr and %expected : !irdl_interp.attr
irdl_interp.assert %result, "Expected %expected"
```

**Note:** Error message quality is deferred to a future phase. Current implementation uses generic messages.

### Constraint Lowering

| Constraint | Generated IR |
|------------|-------------|
| `BaseAttr(T)` | `%check = irdl_interp.check_base %type, "T" : i1`<br>`irdl_interp.assert %check, "Expected T"` |
| `AnyAttr()` | No check (accepts anything) |
| `AnyOf([c1, c2])` | Check base type, `scf.if` dispatch to matching constraint |
| `VarConstraint("T", c)` | Get `%T` from context, check `c` on value, unify, assert |
| `EqAttrConstraint(a)` | `%concrete = irdl_interp.constant.attr a`<br>`%eq = irdl_interp.unify.attr %type, %concrete`<br>`irdl_interp.assert %eq, "Expected a"` |
| `AttrSetConstraint({a, b, c})` | Create constants for each attr, unify with each, OR results, assert |
| `ParamAttrConstraint(T, [p1, p2])` | `%params = irdl_interp.get_params %type`<br>Check each parameter with its constraint |
| `RangeOf(c)` | Extract each element, check with `c` |
| `RangeLengthConstraint(r, n)` | Check length + `RangeOf` |
| `MessageConstraint(c, msg)` | Delegates to `c.emit_irdl_interp()` with `msg` as message prefix |

**MessageConstraint behavior**: Wraps another constraint to add a custom message prefix. When the inner constraint fails, the error message is:
```
msg{inner_constraint_message}
```

Messages typically include a separator like `: ` for readability. This matches Python's `MessageConstraint.verify()` behavior of wrapping exceptions with additional context, but uses simple concatenation instead of newline-based wrapping for simplicity.

### Future Phases

**Not yet implemented:**
- Region shape verification (single-block, etc.)
- Successor verification (signatures, terminators)
- Trait verification (HasParent, HasCanonicalizer, etc.)
- Attribute verification (deprecated path)
- Optional properties

## Next Steps

- Iterate on this design document
- Break first prototype into easy to implement and check parts (maybe type by type/op by op)
- Implement one operation at a time with filecheck tests
- Implement region and block operations (see "Future Phases" above)
- Improve error messages (include actual values in assertion failures)

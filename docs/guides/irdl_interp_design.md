# IRDL Interp Dialect Design

## Motivation

The goal of `irdl_interp` is to provide a runtime for constraint solving in IR, allowing constraint verification, inference, and accessor generation to be expressed as optimizable IR rather than Python code. This mirrors how `pdl_interp` provides primitives for pattern matching.

Instead of using `VerificationContext`/`InferenceContext` (Python implementation artifacts), constraint variables are represented as **SSA values** in the IR, allowing constraint solving to be expressed as dataflow.

## Core Design Principles

1. **Variables as SSA values**: Constraint variables are first-class SSA values, not entries in a dictionary
2. **Unify as the core operation**: A single `unify` operation handles making variables/values equal, resolving or erroring on conflict
3. **Minimal surface area**: Start with the smallest set of operations needed

## Types

### Constraint Variables

```
!irdl_interp.variable.attr   - attribute constraint variable (uninitialized or holds an Attribute)
!irdl_interp.variable.range  - range constraint variable (uninitialized or holds tensor<?x!irdl_interp.attr>)
!irdl_interp.variable.int    - integer constraint variable (uninitialized or holds an index)
```

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

Create an uninitialized constraint variable:

```mlir
%x = irdl_interp.new_var.attr
%y = irdl_interp.new_var.range
%z = irdl_interp.new_var.int
```

**Assembly format**: Type is elided (encoded in operation name).

Variables start uninitialized and can be unified exactly once (subsequent unifies with the same value are idempotent, with different values throw an error).

### Constant Creation

Create concrete attribute values:

```mlir
%attr = irdl_interp.constant.attr i32
```

**Assembly format**: Type is elided. Attribute is passed as an attribute operand.

For integers, use `arith.constant` to create `index` values:

```mlir
%two = arith.constant 2 : index
irdl_interp.unify.int %operand_count, %two : index, index
```

For ranges, use standard tensor operations to construct `tensor<?x!irdl_interp.attr>` values.

### Unify

The core operation. Takes two operands of the same category (attr/range/int). The first operand is always concrete, the second can be a variable or concrete:

```mlir
# Variable with concrete value
irdl_interp.unify.attr %attr : !irdl_interp.attr and %var : !irdl_interp.variable.attr
%int_val = arith.constant 42 : index
irdl_interp.unify.int %int_val : index and %var : !irdl_interp.variable.int
irdl_interp.unify.range %range : tensor<2x!irdl_interp.attr> and %var_range : !irdl_interp.variable.range

# Concrete with concrete (verification)
irdl_interp.unify.attr %attr1 : !irdl_interp.attr and %attr2 : !irdl_interp.attr
irdl_interp.unify.int %count1 : index and %count2 : index
irdl_interp.unify.range %range1 : tensor<2x!irdl_interp.attr> and %range2 : tensor<2x!irdl_interp.attr>
```

**Assembly format**: Each operand's type is printed inline after the operand. This is required because concrete range types (tensors) cannot be inferred from the operation context. Operands separated by `and` keyword.

**Semantics**:
- First unify with a value: sets the variable
- Subsequent unify with same value: succeeds (idempotent)
- Subsequent unify with different value: throws error at runtime
- Variable-variable unify is not supported; reuse the same SSA variable instead

### Check Base

Verify an attribute has a specific base type:

```mlir
irdl_interp.check_base %attr : !irdl_interp.attr, "IntegerType"
irdl_interp.check_base %var : !irdl_interp.variable.attr, "IntegerType"
```

**Assembly format**: Operand type is printed (distinguishes concrete vs variable). Base type name as a bare string attribute. `attr-dict` at end for consistency.

**Semantics**:
- If the operand is a concrete attribute: verifies the base type, throws on mismatch
- If the operand is an unbound variable: no-op (constraint will be checked when variable is bound)
- If the operand is a bound variable: verifies the bound value's base type

## Design Philosophy

Constraint variables should be thought of as **non-backtracking Datalog logic variables**:
- Variables start unbound
- `unify` binds variables or checks equality
- Once bound, variables cannot be rebound to a different value
- Using an unbound variable is undefined behavior
- Unify throws on conflict (unlike Datalog, we don't backtrack - failure is an error)

## Open Questions

1. **Additional accessors**: Do we need access to:
   - Successors? Implemented via `irdl_interp.get_successors` (opaque handles)
   - Regions? Implemented via `irdl_interp.get_regions` (opaque handles)
   - Attribute parameters (for ParamAttrConstraint)? Implemented via `irdl_interp.get_params`
   - Operation attributes (vs properties)? No, these are being deprecated

2. **Attribute operations**: Do we need primitives to:
   - Extract parameters from parametrized attributes? Implemented via `irdl_interp.get_params`
   - Check attribute base type? Implemented via `irdl_interp.check_base`
   - Construct attributes from parameters? I don't think so.

3. **Region and block operations**: Future work needed:
   - Get region body (iterate over operations in a region)
   - Check region properties (single-block, etc.)
   - Get block arguments and terminators
   - Iterate over successor blocks
   - Verify block signatures

## Example: arith.addi verify function

For a complete working example, see the test file:
[`tests/filecheck/dialects/irdl_interp/verify_addi.mlir`](../../tests/filecheck/dialects/irdl_interp/verify_addi.mlir)

This demonstrates:
- Operand/result count verification
- Type constraint checking with shared variables (VarConstraint)
- Property verification (existence, type checking, no extra properties)
- Region and successor count verification

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
| `AnyOf([c1, c2])` | Check each constraint, OR results, assert |
| `VarConstraint("T", c)` | Get `%T` from context, check `c` on value, unify, assert |
| `EqAttrConstraint(a)` | `%concrete = irdl_interp.constant.attr a`<br>`%eq = irdl_interp.unify.attr %type, %concrete`<br>`irdl_interp.assert %eq, "Expected a"` |
| `AttrSetConstraint({a, b, c})` | Create constants for each attr, unify with each, OR results, assert |
| `ParamAttrConstraint(T, [p1, p2])` | `%params = irdl_interp.get_params %type`<br>Check each parameter with its constraint |
| `RangeOf(c)` | Extract each element, check with `c` |
| `RangeLengthConstraint(r, n)` | Check length + `RangeOf` |

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
- Implement `AttrSetConstraint` and `AnyOf` lowering (boolean logic for multiple constraints)
- Add region and block operations:
  - `irdl_interp.get_region_body %region` → iterate over operations
  - `irdl_interp.check_region_single_block %region` → verify single-block property
  - `irdl_interp.get_block_args %block` → get block arguments
  - `irdl_interp.get_block_terminator %block` → get terminator operation
  - `irdl_interp.check_block_signature %block, %expected` → verify signature
- Improve error messages (include actual values in assertion failures)

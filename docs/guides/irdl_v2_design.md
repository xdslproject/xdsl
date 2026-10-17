# IRDL v2 Design

## Motivation

The current IRDL constraint system separates verification and inference into distinct Python methods (`verify` and `infer`). This design proposes a unified constraint solving framework where both verification and inference are expressed as solving a constraint system.

Instead of having separate code paths for:
- **Verification**: Check if values satisfy constraints
- **Inference**: Determine values from constraints when not explicitly provided

Both become queries on the same constraint system:
- **Verification**: "Is this constraint system satisfiable?"
- **Inference**: "What is the solution to this constraint system?"

This mirrors how `irdl_interp` provides a runtime for constraint verification as optimizable IR, but extends it to support full constraint solving including inference.

## Core Design Principles

1. **Expressions over constraints**: Constraints are represented as expressions in a constraint algebra, not as Python objects
2. **Unified solving**: Verification and inference use the same solving mechanism
3. **Deterministic dispatch**: `AnyOf` uses base-type dispatch (not general disjunction) to stay in a tractable fragment
4. **Separation of concerns**: Attribute expressions, range expressions, and integer expressions are separate but composable
5. **Region-based quantification**: `RangeOf` uses region-bearing operations for universal quantification

## Types

### Attribute Expressions

```
!irdl_interp.expr.attr   - attribute constraint expression
```

Attribute expressions represent constraints on attributes. They support:
- Variables (unification targets)
- Constants (exact value constraints)
- Dispatch (base-type selection)

### Range Expressions

```
!irdl_interp.expr.range  - range constraint expression
```

Range expressions represent constraints on sequences of attributes. They support:
- Variables (unification targets)
- Construction from attribute expressions
- Quantified constraints (RangeOf)

### Integer Expressions

```
!irdl_interp.expr.int    - integer constraint expression
```

Integer expressions represent constraints on integers (counts, sizes, etc.). Based on Linear Integer Arithmetic (LIA) - NP-complete in general, but polynomial-time for fixed dimension (n < 10 in our use case).

## Attribute Expression Primitives

### Variables

```mlir
%T = irdl_interp.new_var.attr : !irdl_interp.expr.attr
```

Creates a fresh attribute variable for unification.

### Constants

```mlir
%i32 = irdl_interp.constant.attr i32 : !irdl_interp.expr.attr
```

Lifts a concrete attribute to an expression (for `EqAttrConstraint`).

### Dispatch

```mlir
%dispatch = irdl_interp.dispatch.attr [
  ("IntegerType", %range_expr1),
  ("FloatType", %range_expr2)
] : !irdl_interp.expr.attr
```

Creates a dispatch expression. The dispatch table maps base type names to range expressions. Dispatch is **deterministic** - no overlapping base types allowed.

**Design rationale**: `AnyOf` is not general disjunction (which would require backtracking), but deterministic dispatch based on base type. This keeps the constraint system in a tractable fragment.

## Range Expression Primitives

### Variables

```mlir
%T = irdl_interp.new_var.range : !irdl_interp.expr.range
```

Creates a fresh range variable for unification.

### Build

```mlir
%range = irdl_interp.build.range %a1, %a2, %a3 : !irdl_interp.expr.range
```

Builds a range expression from attribute expressions. Used for `ParamAttrConstraint` where the parameter list is known at compile time.

### Concat

```mlir
%concat = irdl_interp.concat.range %range1, %range2, %range3 : !irdl_interp.expr.range
```

Concatenates multiple range expressions into a single range. This enables uniform length checking - instead of manually verifying operand counts, all operand segments are concatenated and unified with the full operand types in a single operation.

**Example**: For an operation with operands `[fixed1, variadic, fixed2]`:
```mlir
// Build ranges for each segment
%fixed1_range = irdl_interp.build.range %T
%var_range = irdl_interp.rangeof %var_count { ... }
%fixed2_range = irdl_interp.build.range %T

// Concatenate all segments
%all_ranges = irdl_interp.concat.range %fixed1_range, %var_range, %fixed2_range

// Single unification does ALL length checking
%operand_types = irdl_interp.get_operand_types %op
%result = irdl_interp.unify.range %operand_types, %all_ranges : i1
irdl_interp.assert %result, "Operand types must match"
```

**Design rationale**: Length checking becomes implicit in range unification, eliminating the need for manual count verification and segment extraction.

### RangeOf

```mlir
%range = irdl_interp.rangeof %count {
^bb0(%elem: !irdl_interp.expr.attr):
  %result = irdl_interp.unify.attr %elem, %T : i1
  irdl_interp.assert %result, "Element must satisfy T"
} : !irdl_interp.expr.range
```

Creates a constrained range expression. The region body defines the constraint on each element. The `%count` operand specifies the expected length (can be an integer expression).

**Design rationale**: `RangeOf` uses a region to express universal quantification ("for all elements, this constraint holds"). This is more general than a simple `RangeOf(attr_constraint)` and enables complex per-element constraints.

## Integer Expression Primitives

See `irdl_interp_design.md` for the complete integer expression algebra. In summary:
- `irdl_interp.new_var.int` - Variable
- `irdl_interp.constant.int` - Constant
- `irdl_interp.add.int` - Addition
- `irdl_interp.scalar_mul.int` - Scalar multiplication
- `irdl_interp.unify.int` - Equality solving (expr1 = expr2)
- `irdl_interp.le.int` - Inequality checking (expr1 <= expr2)

## Operations

### Unify

Unify operations add constraints to the system and return `i1` indicating satisfiability.

```mlir
// Attribute unification
%result = irdl_interp.unify.attr %attr_expr1, %attr_expr2 : i1

// Range unification
%result = irdl_interp.unify.range %range_expr1, %range_expr2 : i1

// Integer unification
%result = irdl_interp.unify.int %int_expr1, %int_expr2 : i1
```

**Semantics**:
- Returns `true` if the constraint is satisfiable
- Returns `false` if there is an explicit contradiction
- Does **not** fail if variables remain undersolved (checked later by `read.*` operations)
- Non-failing: use `irdl_interp.assert` to enforce constraints

**Dispatch unification**: When unifying an attribute with a dispatch expression:
1. Look up the attribute's base type in the dispatch table
2. If no match: return `false`
3. If match: unify the attribute with the range expression from the matching branch

### Assert

```mlir
irdl_interp.assert %condition, "Error message"
```

Asserts that a condition is true. Fails verification with the given message if false.

### Read (Phase 2)

```mlir
%attr_val = irdl_interp.read.attr %attr_expr : !irdl_interp.attr
%range_val = irdl_interp.read.range %range_expr : tensor<?x!irdl_interp.attr>
%int_val = irdl_interp.read.int %int_expr : index
```

Extracts concrete values from solved expressions. Fails if the expression is undersolved.

**Design rationale**: Reading is separate from solving. This allows the solver to work with undersolved expressions during constraint propagation, and only fail when a concrete value is actually needed.

## Examples

### Example 1: Simple Type Equality (VarConstraint)

```python
# Python IRDL
T = VarConstraint("T", AnyAttr())
lhs = operand_def(T)
rhs = operand_def(T)
result = result_def(T)
```

```mlir
# IRDL v2
%T = irdl_interp.new_var.attr
%lhs_type = tensor.extract %operand_types[0]
%rhs_type = tensor.extract %operand_types[1]
%result_type = tensor.extract %result_types[0]

%r1 = irdl_interp.unify.attr %lhs_type, %T : i1
%r2 = irdl_interp.unify.attr %rhs_type, %T : i1
%r3 = irdl_interp.unify.attr %result_type, %T : i1

irdl_interp.assert %r1, "lhs type must match T"
irdl_interp.assert %r2, "rhs type must match T"
irdl_interp.assert %r3, "result type must match T"
```

### Example 2: Base Type Constraint (AnyOf via Dispatch)

```python
# Python IRDL
T = VarConstraint("T", IntegerType | FloatType)
```

```mlir
# IRDL v2
%T = irdl_interp.new_var.attr
%dispatch = irdl_interp.dispatch.attr [
  ("IntegerType", %int_range),
  ("FloatType", %float_range)
]

%result = irdl_interp.unify.attr %T, %dispatch : i1
irdl_interp.assert %result, "T must be IntegerType or FloatType"
```

### Example 3: Parametrized Attribute (ParamAttrConstraint)

```python
# Python IRDL
T = VarConstraint("T", ParamAttrConstraint(VectorType, [BaseAttr(IntegerType)]))
```

```mlir
# IRDL v2
%T = irdl_interp.new_var.attr
%param_range = irdl_interp.build.range %int_constraint
%dispatch = irdl_interp.dispatch.attr [
  ("VectorType", %param_range)
]

%result = irdl_interp.unify.attr %T, %dispatch : i1
irdl_interp.assert %result, "T must be VectorType"
```

### Example 4: Variadic Operands (RangeOf)

```python
# Python IRDL
T = VarConstraint("T", AnyAttr())
args = var_operand_def(T)  # Variadic: all args have type T
```

```mlir
# IRDL v2
%T = irdl_interp.new_var.attr
%arg_count = tensor.dim %operand_types, %c0

%T_range = irdl_interp.rangeof %arg_count {
^bb0(%elem: !irdl_interp.expr.attr):
  %result = irdl_interp.unify.attr %elem, %T : i1
  irdl_interp.assert %result, "Element must match T"
}

%result = irdl_interp.unify.range %operand_types, %T_range : i1
irdl_interp.assert %result, "All args must satisfy constraint"
```

### Example 5: Mixed Constraints (varith.add)

```python
# Python IRDL
T = VarConstraint("T", IntegerType | IndexType | FloatType)
args = var_operand_def(T)  # Variadic, all same type
result = result_def(T)
```

```mlir
# IRDL v2
%T = irdl_interp.new_var.attr
%arg_count = tensor.dim %operand_types, %c0

// Dispatch for AnyOf constraint
%dispatch = irdl_interp.dispatch.attr [
  ("IntegerType", %int_range),
  ("IndexType", %index_range),
  ("FloatType", %float_range)
]

// RangeOf with dispatch in region
%T_range = irdl_interp.rangeof %arg_count {
^bb0(%elem: !irdl_interp.expr.attr):
  %result = irdl_interp.unify.attr %elem, %dispatch : i1
  irdl_interp.assert %result, "Element must be IntegerType, IndexType, or FloatType"
}

// Unify operand types with constrained range
%result = irdl_interp.unify.range %operand_types, %T_range : i1
irdl_interp.assert %result, "All args must satisfy constraint"

// Verify result type matches
%result_type = tensor.extract %result_types[%c0]
%r2 = irdl_interp.unify.attr %result_type, %T : i1
irdl_interp.assert %r2, "Result type must match T"
```

## Relationship to IRDL Interp

IRDL v2 **extends** `irdl_interp` with:
- Attribute expressions (variables, constants, dispatch)
- Range expressions (variables, build, rangeof)
- Region-bearing operations (rangeof)
- Unified solving semantics

The integer expression algebra from `irdl_interp` is reused as-is.

**Lowering strategy**: IRDL v2 constraints are lowered to optimized `irdl_interp` expressions by **performing constraint solving during lowering**. Instead of generating generic constraint-solving IR, the lowering pass:

1. Analyzes the constraint system
2. Solves what can be determined statically
3. Emits specialized `irdl_interp` IR with minimal solving overhead

For example, a `RangeOf` with a simple constraint might lower to an explicit loop with direct unifications, rather than a generic range-solving operation. This enables the `irdl_interp` canonicalizer to further optimize the generated IR.

## Open Questions

1. **Range expression algebra**: Is `build.range` sufficient, or do we need concatenation (`concat.range`)?

2. **RangeOf region semantics**: Should the region have explicit yield, or are unifies/asserts inside the region sufficient?

3. **Dispatch table construction**: How is the dispatch table built from Python IRDL constraints? Is it generated during lowering, or constructed at runtime?

4. **TypeVarConstraint**: How do Python `TypeVar` constraints map to the expression algebra? Are they just variables, or is there special handling?

5. **SizedConstraint**: How do we constrain the length/shape of sized attributes? Does this bridge attribute and integer expressions?

6. **Solver implementation**: Should the solver be:
   - Pure Python in the interpreter?
   - MLIR canonicalization patterns?
   - Hybrid (simple cases in MLIR, complex in Python)?

7. **Error messages**: How do we generate helpful error messages from constraint solving failures? Current design uses `irdl_interp.assert` with messages, but solving failures may be more complex.

8. **Inference API**: How does Python code query solved values? Is there a `read.*` operation, or is inference handled differently?

9. **Backward compatibility**: How does IRDL v2 coexist with current IRDL? Is it a clean break, or is there a migration path?

10. **Performance**: What is the performance characteristic of constraint solving vs. current verify/infer? Are there cases where solving is significantly slower?

## See Also

- `irdl_interp_design.md` - Integer expression algebra and solving semantics
- `constraints.md` - Current IRDL constraint system (for comparison)

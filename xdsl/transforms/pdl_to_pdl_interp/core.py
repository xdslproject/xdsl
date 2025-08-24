from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, cast

# =============================================================================
# Core Data Structures - Positions
# =============================================================================


class PositionKind(Enum):
    OPERATION = "operation"
    OPERAND = "operand"
    OPERAND_GROUP = "operand_group"
    RESULT = "result"
    RESULT_GROUP = "result_group"
    ATTRIBUTE = "attribute"
    TYPE = "type"
    USERS = "users"


@dataclass
class Position(ABC):
    """Base class for all position types"""

    kind: PositionKind
    parent: Optional["Position"] = None

    def get_operation_depth(self) -> int:
        """Returns depth of first ancestor operation position"""
        if isinstance(self, OperationPosition):
            return self.depth
        return self.parent.get_operation_depth() if self.parent else 0


@dataclass
class OperationPosition(Position):
    """Represents an operation in the IR"""

    depth: int = 0

    def __post_init__(self):
        super().__init__(PositionKind.OPERATION, self.parent)

    def is_root(self) -> bool:
        return self.depth == 0

    def is_operand_defining_op(self) -> bool:
        return isinstance(self.parent, (OperandPosition | OperandGroupPosition))


@dataclass
class OperandPosition(Position):
    """Represents an operand of an operation"""

    operand_number: int = -1

    def __post_init__(self):
        super().__init__(PositionKind.OPERAND, self.parent)


@dataclass
class OperandGroupPosition(Position):
    """Represents a group of operands"""

    group_number: int | None = None
    is_variadic: bool = False

    def __post_init__(self):
        super().__init__(PositionKind.OPERAND_GROUP, self.parent)


@dataclass
class ResultPosition(Position):
    """Represents a result of an operation"""

    result_number: int = -1

    def __post_init__(self):
        super().__init__(PositionKind.RESULT, self.parent)


@dataclass
class AttributePosition(Position):
    """Represents an attribute of an operation"""

    attribute_name: str = ""

    def __post_init__(self):
        super().__init__(PositionKind.ATTRIBUTE, self.parent)


@dataclass
class TypePosition(Position):
    """Represents the type of a value"""

    def __post_init__(self):
        super().__init__(PositionKind.TYPE, self.parent)


@dataclass
class UsersPosition(Position):
    """Represents users of a value"""

    use_representative: bool = False

    def __post_init__(self):
        super().__init__(PositionKind.USERS, self.parent)


# =============================================================================
# Predicate System - Questions and Answers
# =============================================================================


class PredicateKind(Enum):
    # Questions
    IS_NOT_NULL = "is_not_null"
    OPERATION_NAME = "operation_name"
    OPERAND_COUNT = "operand_count"
    RESULT_COUNT = "result_count"
    TYPE_CONSTRAINT = "type_constraint"
    ATTRIBUTE_CONSTRAINT = "attribute_constraint"
    EQUAL_TO = "equal_to"

    # Answers
    TRUE = "true"
    FALSE = "false"
    UNSIGNED = "unsigned"
    STRING = "string"
    TYPE = "type"
    ATTRIBUTE = "attribute"


@dataclass
class Predicate(ABC):
    """Base predicate class"""

    kind: PredicateKind


@dataclass
class Question(Predicate):
    """Represents a question/check to perform"""

    pass


@dataclass
class Answer(Predicate):
    """Represents an expected answer to a question"""

    value: Any = None


# Specific Question Types
@dataclass
class IsNotNullQuestion(Question):
    def __post_init__(self):
        super().__init__(PredicateKind.IS_NOT_NULL)


@dataclass
class OperationNameQuestion(Question):
    def __post_init__(self):
        super().__init__(PredicateKind.OPERATION_NAME)


@dataclass
class OperandCountQuestion(Question):
    def __post_init__(self):
        super().__init__(PredicateKind.OPERAND_COUNT)


@dataclass
class ResultCountQuestion(Question):
    def __post_init__(self):
        super().__init__(PredicateKind.RESULT_COUNT)


@dataclass
class EqualToQuestion(Question):
    other_position: Position

    def __post_init__(self):
        super().__init__(PredicateKind.EQUAL_TO)


# Answer Types
@dataclass
class TrueAnswer(Answer):
    def __post_init__(self):
        super().__init__(PredicateKind.TRUE, True)


@dataclass
class UnsignedAnswer(Answer):
    def __post_init__(self):
        super().__init__(PredicateKind.UNSIGNED, self.value)


@dataclass
class StringAnswer(Answer):
    def __post_init__(self):
        super().__init__(PredicateKind.STRING, self.value)


# =============================================================================
# Positional Predicates
# =============================================================================


@dataclass
class PositionalPredicate:
    """A predicate applied to a specific position"""

    position: Position
    question: Question
    answer: Answer


# =============================================================================
# Predicate Builder
# =============================================================================


class PredicateBuilder:
    """Utility for constructing predicates and positions"""

    def __init__(self):
        self._position_cache: dict[tuple, Position] = {}

    def get_root(self) -> OperationPosition:
        """Get the root operation position"""
        key = ("root",)
        if key not in self._position_cache:
            self._position_cache[key] = OperationPosition(depth=0, parent=None)
        return self._position_cache[key]

    def get_operand_defining_op(self, operand_pos: Position) -> OperationPosition:
        """Get the operation that defines an operand"""
        key = ("defining_op", operand_pos)
        if key not in self._position_cache:
            depth = operand_pos.get_operation_depth() + 1
            self._position_cache[key] = OperationPosition(
                depth=depth, parent=operand_pos
            )
        return self._position_cache[key]

    def get_operand(
        self, op_pos: OperationPosition, operand_num: int
    ) -> OperandPosition:
        """Get an operand position"""
        key = ("operand", op_pos, operand_num)
        if key not in self._position_cache:
            self._position_cache[key] = OperandPosition(
                operand_number=operand_num, parent=op_pos
            )
        return self._position_cache[key]

    def get_result(self, op_pos: OperationPosition, result_num: int) -> ResultPosition:
        """Get a result position"""
        key = ("result", op_pos, result_num)
        if key not in self._position_cache:
            self._position_cache[key] = ResultPosition(
                result_number=result_num, parent=op_pos
            )
        return self._position_cache[key]

    def get_attribute(
        self, op_pos: OperationPosition, attr_name: str
    ) -> AttributePosition:
        """Get an attribute position"""
        key = ("attribute", op_pos, attr_name)
        if key not in self._position_cache:
            self._position_cache[key] = AttributePosition(
                attribute_name=attr_name, parent=op_pos
            )
        return self._position_cache[key]

    def get_type(self, pos: Position) -> TypePosition:
        """Get a type position"""
        key = ("type", pos)
        if key not in self._position_cache:
            self._position_cache[key] = TypePosition(parent=pos)
        return self._position_cache[key]

    # Predicate builders
    def get_is_not_null(self) -> tuple[Question, Answer]:
        return (IsNotNullQuestion(), TrueAnswer())

    def get_operation_name(self, name: str) -> tuple[Question, Answer]:
        return (OperationNameQuestion(), StringAnswer(value=name))

    def get_operand_count(self, count: int) -> tuple[Question, Answer]:
        return (OperandCountQuestion(), UnsignedAnswer(value=count))

    def get_result_count(self, count: int) -> tuple[Question, Answer]:
        return (ResultCountQuestion(), UnsignedAnswer(value=count))

    def get_equal_to(self, other_pos: Position) -> tuple[Question, Answer]:
        return (EqualToQuestion(other_position=other_pos), TrueAnswer())


# =============================================================================
# Matcher Tree Nodes
# =============================================================================


@dataclass
class MatcherNode(ABC):
    """Base class for matcher tree nodes"""

    position: Position | None = None
    question: Question | None = None
    failure_node: Optional["MatcherNode"] = None


@dataclass
class BoolNode(MatcherNode):
    """Boolean predicate node"""

    answer: Answer = None
    success_node: MatcherNode | None = None


@dataclass
class SwitchNode(MatcherNode):
    """Multi-way switch node"""

    children: dict[Answer, MatcherNode] = field(default_factory=dict)


@dataclass
class SuccessNode(MatcherNode):
    """Successful pattern match"""

    pattern: Any = None  # PDL pattern reference
    root: Any = None  # Root value


@dataclass
class ExitNode(MatcherNode):
    """Exit/failure node"""

    pass


# =============================================================================
# Root Ordering and Cost Graph
# =============================================================================


@dataclass
class RootOrderingEntry:
    """Entry in the root ordering cost graph"""

    cost: tuple[int, int]  # (depth, tie_breaker)
    connector: Any  # Value that connects the roots


class OptimalBranching:
    """Edmonds' optimal branching algorithm for minimum spanning arborescence"""

    def __init__(self, graph: dict[Any, dict[Any, RootOrderingEntry]], root: Any):
        self.graph = graph
        self.root = root
        self.parents: dict[Any, Any] = {}

    def solve(self) -> int:
        """Solve for optimal branching, returns total cost"""
        self.parents.clear()
        self.parents[self.root] = None
        total_cost = 0

        # Find minimum incoming edge for each node
        for target in self.graph:
            if target in self.parents:
                continue

            # Follow chain of minimum parents
            node = target
            parent_depths: dict[Any, int] = {}

            while node not in self.parents:
                if node not in self.graph:
                    break

                # Find best parent
                best_parent = None
                best_cost = None

                for source, entry in self.graph[node].items():
                    if best_parent is None or entry.cost < cast(
                        tuple[int, int], best_cost
                    ):
                        best_parent = source
                        best_cost = entry.cost

                if best_parent is None:
                    break
                assert best_cost is not None

                self.parents[node] = best_parent
                parent_depths[node] = best_cost[0]
                total_cost += best_cost[0]
                node = best_parent

            # Check for cycles and contract if needed
            if node in parent_depths:
                cycle = self._get_cycle(node)
                total_cost += self._contract_cycle(cycle, parent_depths)

        return total_cost

    def _get_cycle(self, start: Any) -> list[Any]:
        """Get cycle starting from the given node"""
        cycle: list[Any] = []
        node = start
        while True:
            cycle.append(node)
            node = self.parents[node]
            if node == start:
                break
        return cycle

    def _contract_cycle(self, cycle: list[Any], parent_depths: dict[Any, int]) -> int:
        """Contract a cycle in the graph"""
        # Simplified cycle contraction
        cycle_cost = sum(parent_depths[node] for node in cycle)

        # Update parents to break cycle
        _rep = cycle[0]
        for i, node in enumerate(cycle[:-1]):
            self.parents[node] = cycle[i + 1]

        return cycle_cost


# =============================================================================
# Pattern Analysis
# =============================================================================


class PatternAnalyzer:
    """Analyzes PDL patterns and extracts predicates"""

    def __init__(self, builder: PredicateBuilder):
        self.builder = builder

    def detect_roots(self, pattern) -> list[Any]:
        """Detect root operations in a pattern (pseudo-code interface)"""
        # PSEUDO: Find operations whose results aren't consumed by other ops
        roots = []
        used_ops = set()

        for op in pattern.get_operations():
            for operand in op.get_operands():
                if hasattr(operand, "defining_op"):
                    used_ops.add(operand.defining_op)

        for op in pattern.get_operations():
            if op not in used_ops:
                roots.append(op)

        return roots

    def extract_tree_predicates(
        self, value, position: Position, inputs: dict[Any, Position]
    ) -> list[PositionalPredicate]:
        """Extract predicates by walking the operation tree"""
        predicates = []

        # Avoid revisiting values
        if value in inputs:
            if inputs[value] != position:
                # Add equality constraint
                q, a = self.builder.get_equal_to(inputs[value])
                predicates.append(PositionalPredicate(position, q, a))
            return predicates

        inputs[value] = position

        # PSEUDO: Interface to PDL IR
        if value.is_operation():
            predicates.extend(
                self._extract_operation_predicates(value, position, inputs)
            )
        elif value.is_operand():
            predicates.extend(self._extract_operand_predicates(value, position, inputs))

        return predicates

    def _extract_operation_predicates(
        self, op_value, op_pos: OperationPosition, inputs: dict[Any, Position]
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operation"""
        predicates = []

        # PSEUDO: Access PDL operation properties
        if not op_pos.is_root():
            q, a = self.builder.get_is_not_null()
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Operation name check
        if op_name := op_value.get_operation_name():
            q, a = self.builder.get_operation_name(op_name)
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Operand count check
        operand_count = op_value.get_operand_count()
        q, a = self.builder.get_operand_count(operand_count)
        predicates.append(PositionalPredicate(op_pos, q, a))

        # Result count check
        result_count = op_value.get_result_count()
        q, a = self.builder.get_result_count(result_count)
        predicates.append(PositionalPredicate(op_pos, q, a))

        # Recurse into operands
        for i, operand in enumerate(op_value.get_operands()):
            operand_pos = self.builder.get_operand(op_pos, i)
            predicates.extend(
                self.extract_tree_predicates(operand, operand_pos, inputs)
            )

        return predicates

    def _extract_operand_predicates(
        self, operand_value, operand_pos: OperandPosition, inputs: dict[Any, Position]
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operand"""
        predicates = []

        # Not null check
        q, a = self.builder.get_is_not_null()
        predicates.append(PositionalPredicate(operand_pos, q, a))

        # If operand has a defining operation, recurse
        if defining_op := operand_value.get_defining_operation():
            def_op_pos = self.builder.get_operand_defining_op(operand_pos)
            predicates.extend(
                self.extract_tree_predicates(defining_op, def_op_pos, inputs)
            )

        return predicates


# =============================================================================
# Predicate Ordering and Tree Construction
# =============================================================================


@dataclass
class OrderedPredicate:
    """Predicate with ordering information"""

    position: Position
    question: Question
    primary_score: int = 0  # Frequency across patterns
    secondary_score: int = 0  # Squared sum within patterns
    tie_breaker: int = 0  # Insertion order
    pattern_answers: dict[Any, Answer] = field(default_factory=lambda: {})

    def __lt__(self, other: "OrderedPredicate") -> bool:
        """Comparison for priority ordering"""
        return (
            self.primary_score,
            self.secondary_score,
            -self.position.get_operation_depth(),  # Prefer lower depth
            -self.position.kind.value,  # Position dependency
            -self.question.kind.value,  # Predicate dependency
            -self.tie_breaker,  # Deterministic order
        ) > (
            other.primary_score,
            other.secondary_score,
            -other.position.get_operation_depth(),
            -other.position.kind.value,
            -other.question.kind.value,
            -other.tie_breaker,
        )


class PredicateTreeBuilder:
    """Builds optimized predicate matching trees"""

    def __init__(self):
        self.analyzer = PatternAnalyzer(PredicateBuilder())

    def build_predicate_tree(self, patterns: list[Any]) -> MatcherNode:
        """Build optimized matcher tree from multiple patterns"""

        # Extract predicates from all patterns
        all_pattern_predicates = []
        for pattern in patterns:
            predicates = self._extract_pattern_predicates(pattern)
            all_pattern_predicates.append((pattern, predicates))

        # Create ordered predicates with frequency analysis
        ordered_predicates = self._create_ordered_predicates(all_pattern_predicates)

        # Sort predicates by priority
        sorted_predicates = sorted(ordered_predicates.values())

        # Build matcher tree by propagating patterns
        root_node = None
        for pattern, predicates in all_pattern_predicates:
            pattern_predicate_set = {
                (pred.position, pred.question): pred for pred in predicates
            }
            root_node = self._propagate_pattern(
                root_node, pattern, pattern_predicate_set, sorted_predicates, 0
            )

        # Add exit node and optimize
        self._insert_exit_node(root_node)
        self._optimize_tree(root_node)

        return root_node

    def _extract_pattern_predicates(self, pattern) -> list[PositionalPredicate]:
        """Extract all predicates for a single pattern"""
        roots = self.analyzer.detect_roots(pattern)

        # For simplicity, use first root (in real implementation,
        # would use optimal root selection)
        root = roots[0] if roots else None
        if not root:
            return []

        inputs = {}
        root_pos = self.analyzer.builder.get_root()

        return self.analyzer.extract_tree_predicates(root, root_pos, inputs)

    def _create_ordered_predicates(
        self, all_pattern_predicates
    ) -> dict[tuple, OrderedPredicate]:
        """Create ordered predicates with frequency analysis"""
        predicate_map = {}
        tie_breaker = 0

        # Collect unique predicates
        for pattern, predicates in all_pattern_predicates:
            for pred in predicates:
                key = (pred.position, pred.question)

                if key not in predicate_map:
                    ordered_pred = OrderedPredicate(
                        position=pred.position,
                        question=pred.question,
                        tie_breaker=tie_breaker,
                    )
                    predicate_map[key] = ordered_pred
                    tie_breaker += 1

                # Track pattern answers and increment frequency
                predicate_map[key].pattern_answers[pattern] = pred.answer
                predicate_map[key].primary_score += 1

        # Calculate secondary scores
        for pattern, predicates in all_pattern_predicates:
            pattern_primary_sum = 0
            pattern_predicates = []

            for pred in predicates:
                key = (pred.position, pred.question)
                ordered_pred = predicate_map[key]
                pattern_predicates.append(ordered_pred)
                pattern_primary_sum += ordered_pred.primary_score**2

            # Add to secondary score
            for ordered_pred in pattern_predicates:
                ordered_pred.secondary_score += pattern_primary_sum

        return predicate_map

    def _propagate_pattern(
        self,
        node: MatcherNode | None,
        pattern: Any,
        pattern_predicates: dict[tuple, PositionalPredicate],
        sorted_predicates: list[OrderedPredicate],
        predicate_index: int,
    ) -> MatcherNode:
        """Propagate a pattern through the predicate tree"""

        # Base case: reached end of predicates
        if predicate_index >= len(sorted_predicates):
            return SuccessNode(pattern=pattern, root=None, failure_node=node)

        current_predicate = sorted_predicates[predicate_index]
        pred_key = (current_predicate.position, current_predicate.question)

        # Skip predicates not in this pattern
        if pred_key not in pattern_predicates:
            return self._propagate_pattern(
                node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index + 1,
            )

        # Create or match existing node
        if node is None:
            # Create new switch node
            node = SwitchNode(
                position=current_predicate.position, question=current_predicate.question
            )

        if self._nodes_match(node, current_predicate):
            # Continue down matching path
            pattern_answer = pattern_predicates[pred_key].answer

            if isinstance(node, SwitchNode):
                if pattern_answer not in node.children:
                    node.children[pattern_answer] = None

                node.children[pattern_answer] = self._propagate_pattern(
                    node.children[pattern_answer],
                    pattern,
                    pattern_predicates,
                    sorted_predicates,
                    predicate_index + 1,
                )

        else:
            # Divergence - continue down failure path
            node.failure_node = self._propagate_pattern(
                node.failure_node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index,
            )

        return node

    def _nodes_match(self, node: MatcherNode, predicate: OrderedPredicate) -> bool:
        """Check if node matches the given predicate"""
        return (
            node.position == predicate.position and node.question == predicate.question
        )

    def _insert_exit_node(self, root: MatcherNode):
        """Insert exit node at end of failure paths"""

        def insert_exit_recursive(node):
            if node is None:
                return ExitNode()
            if node.failure_node is None:
                node.failure_node = ExitNode()
            else:
                node.failure_node = insert_exit_recursive(node.failure_node)
            return node

        return insert_exit_recursive(root)

    def _optimize_tree(self, root: MatcherNode):
        """Optimize the tree by collapsing single-child switches to bools"""
        if isinstance(root, SwitchNode) and len(root.children) == 1:
            # Convert switch to bool node
            answer, child = next(iter(root.children.items()))
            bool_node = BoolNode(
                position=root.position,
                question=root.question,
                answer=answer,
                success_node=child,
                failure_node=root.failure_node,
            )
            return bool_node

        # Recursively optimize children
        if isinstance(root, SwitchNode):
            for answer in root.children:
                root.children[answer] = self._optimize_tree(root.children[answer])
        elif isinstance(root, BoolNode):
            root.success_node = self._optimize_tree(root.success_node)

        if root.failure_node:
            root.failure_node = self._optimize_tree(root.failure_node)

        return root


# =============================================================================
# Code Generation
# =============================================================================


class PDLInterpCodeGenerator:
    """Generates pdl_interp code from matcher trees"""

    def __init__(self):
        self.block_counter = 0
        self.value_counter = 0

    def generate_matcher_function(
        self, root: MatcherNode, name: str = "matcher"
    ) -> str:
        """Generate complete pdl_interp matcher function"""
        self.block_counter = 0
        self.value_counter = 0

        code = [f"pdl_interp.func @{name}(%arg0: !pdl.operation) {{"]

        # Generate body
        entry_block = self._generate_matcher_code(root, "entry")
        code.extend(f"  {line}" for line in entry_block)

        code.append("}")
        return "\n".join(code)

    def _generate_matcher_code(self, node: MatcherNode, block_name: str) -> list[str]:
        """Generate code for a matcher node"""
        if node is None:
            return []

        if isinstance(node, ExitNode):
            return ["pdl_interp.finalize"]

        elif isinstance(node, SuccessNode):
            code = []
            # Record successful match
            code.append(
                "pdl_interp.record_match @rewriters::@pdl_generated_rewriter(...)"
            )

            # Continue to failure node
            if node.failure_node:
                failure_code = self._generate_matcher_code(
                    node.failure_node, f"^bb{self.block_counter}"
                )
                code.extend(failure_code)

            return code

        elif isinstance(node, BoolNode):
            return self._generate_bool_node_code(node)

        elif isinstance(node, SwitchNode):
            return self._generate_switch_node_code(node)

        return []

    def _generate_bool_node_code(self, node: BoolNode) -> list[str]:
        """Generate code for boolean predicate node"""
        code = []

        # Get value at position
        value_access = self._generate_value_access(node.position)
        code.extend(value_access["setup"])
        current_value = value_access["result"]

        # Generate predicate check
        success_block = f"^bb{self.block_counter}"
        self.block_counter += 1
        failure_block = f"^bb{self.block_counter}"
        self.block_counter += 1

        predicate_code = self._generate_predicate_check(
            node.question, node.answer, current_value, success_block, failure_block
        )
        code.extend(predicate_code)

        # Success block
        code.append(f"{success_block}:")
        success_code = self._generate_matcher_code(node.success_node, success_block)
        code.extend(f"  {line}" for line in success_code)

        # Failure block
        code.append(f"{failure_block}:")
        failure_code = self._generate_matcher_code(node.failure_node, failure_block)
        code.extend(f"  {line}" for line in failure_code)

        return code

    def _generate_switch_node_code(self, node: SwitchNode) -> list[str]:
        """Generate code for switch node"""
        code = []

        # Get value at position
        value_access = self._generate_value_access(node.position)
        code.extend(value_access["setup"])
        current_value = value_access["result"]

        # Generate switch operation
        default_block = f"^bb{self.block_counter}"
        self.block_counter += 1

        case_blocks = []
        case_values = []

        for answer, child_node in node.children.items():
            case_block = f"^bb{self.block_counter}"
            self.block_counter += 1
            case_blocks.append(case_block)
            case_values.append(self._answer_to_string(answer))

        # Generate switch instruction
        switch_op = self._generate_switch_operation(
            node.question, current_value, case_values, default_block, case_blocks
        )
        code.extend(switch_op)

        # Generate case blocks
        for i, (answer, child_node) in enumerate(node.children.items()):
            code.append(f"{case_blocks[i]}:")
            child_code = self._generate_matcher_code(child_node, case_blocks[i])
            code.extend(f"  {line}" for line in child_code)

        # Default/failure block
        code.append(f"{default_block}:")
        failure_code = self._generate_matcher_code(node.failure_node, default_block)
        code.extend(f"  {line}" for line in failure_code)

        return code

    def _generate_value_access(self, position: Position) -> dict[str, Any]:
        """Generate code to access value at position"""
        setup_code = []

        if isinstance(position, OperationPosition):
            if position.is_root():
                return {"setup": [], "result": "%arg0"}
            else:
                # Get defining op of parent
                parent_access = self._generate_value_access(position.parent)
                setup_code.extend(parent_access["setup"])

                result_val = f"%{self.value_counter}"
                self.value_counter += 1
                setup_code.append(
                    f"{result_val} = pdl_interp.get_defining_op of {parent_access['result']} : !pdl.operation"
                )
                return {"setup": setup_code, "result": result_val}

        elif isinstance(position, OperandPosition):
            parent_access = self._generate_value_access(position.parent)
            setup_code.extend(parent_access["setup"])

            result_val = f"%{self.value_counter}"
            self.value_counter += 1
            setup_code.append(
                f"{result_val} = pdl_interp.get_operand {position.operand_number} of {parent_access['result']}"
            )
            return {"setup": setup_code, "result": result_val}

        elif isinstance(position, ResultPosition):
            parent_access = self._generate_value_access(position.parent)
            setup_code.extend(parent_access["setup"])

            result_val = f"%{self.value_counter}"
            self.value_counter += 1
            setup_code.append(
                f"{result_val} = pdl_interp.get_result {position.result_number} of {parent_access['result']}"
            )
            return {"setup": setup_code, "result": result_val}

        # Add other position types as needed
        return {"setup": setup_code, "result": f"%unknown_{self.value_counter}"}

    def _generate_predicate_check(
        self,
        question: Question,
        answer: Answer,
        value: str,
        success_block: str,
        failure_block: str,
    ) -> list[str]:
        """Generate predicate check operation"""
        if isinstance(question, IsNotNullQuestion):
            return [
                f"pdl_interp.is_not_null {value} -> {success_block}, {failure_block}"
            ]

        elif isinstance(question, OperationNameQuestion):
            op_name = answer.value
            return [
                f'pdl_interp.check_operation_name of {value} is "{op_name}" -> {success_block}, {failure_block}'
            ]

        elif isinstance(question, OperandCountQuestion):
            count = answer.value
            return [
                f"pdl_interp.check_operand_count of {value} is {count} -> {success_block}, {failure_block}"
            ]

        elif isinstance(question, ResultCountQuestion):
            count = answer.value
            return [
                f"pdl_interp.check_result_count of {value} is {count} -> {success_block}, {failure_block}"
            ]

        elif isinstance(question, EqualToQuestion):
            other_access = self._generate_value_access(question.other_position)
            return other_access["setup"] + [
                f"pdl_interp.are_equal {value}, {other_access['result']} -> {success_block}, {failure_block}"
            ]

        return [f"// Unknown predicate check for {value}"]

    def _generate_switch_operation(
        self,
        question: Question,
        value: str,
        case_values: list[str],
        default_block: str,
        case_blocks: list[str],
    ) -> list[str]:
        """Generate switch operation"""
        if isinstance(question, OperationNameQuestion):
            values_str = ", ".join(f'"{v}"' for v in case_values)
            blocks_str = ", ".join(case_blocks)
            return [
                f"pdl_interp.switch_operation_name {value} [{values_str}] -> {default_block}, [{blocks_str}]"
            ]

        elif isinstance(question, OperandCountQuestion):
            values_str = ", ".join(case_values)
            blocks_str = ", ".join(case_blocks)
            return [
                f"pdl_interp.switch_operand_count {value} [{values_str}] -> {default_block}, [{blocks_str}]"
            ]

        return [f"// Unknown switch operation for {value}"]

    def _answer_to_string(self, answer: Answer) -> str:
        """Convert answer to string representation"""
        return str(answer.value) if answer.value is not None else "null"


# =============================================================================
# Main Transformation Pipeline
# =============================================================================


class PDLToPDLInterpTransformer:
    """Main transformer class that orchestrates the entire conversion"""

    def __init__(self):
        self.tree_builder = PredicateTreeBuilder()
        self.code_generator = PDLInterpCodeGenerator()

    def transform(self, pdl_patterns: list[Any]) -> str:
        """Transform PDL patterns to pdl_interp code"""

        print("=== PDL to PDL_interp Transformation ===")
        print(f"Processing {len(pdl_patterns)} patterns...")

        # Stage 1: Build predicate tree
        print("\nStage 1: Building predicate tree...")
        matcher_tree = self.tree_builder.build_predicate_tree(pdl_patterns)

        # Stage 2: Generate code
        print("Stage 2: Generating pdl_interp code...")
        generated_code = self.code_generator.generate_matcher_function(matcher_tree)

        print("Stage 3: Transformation complete!\n")
        return generated_code


# =============================================================================
# Example Usage and Demo
# =============================================================================


def demo_transformation():
    """Demonstrate the transformation with a simple example"""

    # Mock PDL pattern (pseudo-code representation)
    class MockPDLPattern:
        def __init__(self, name):
            self.name = name

        def get_operations(self):
            # Simplified representation
            return [
                MockOperation("arith.mulf"),
                MockOperation("arith.absf"),
                MockOperation("math.sin"),
            ]

    class MockOperation:
        def __init__(self, name):
            self.name = name

        def get_operation_name(self):
            return self.name

        def get_operand_count(self):
            return 2 if self.name == "arith.mulf" else 1

        def get_result_count(self):
            return 1

        def get_operands(self):
            return []  # Simplified

    # Create transformer
    transformer = PDLToPDLInterpTransformer()

    # Mock patterns
    patterns = [MockPDLPattern("example_pattern")]

    # Transform
    result = transformer.transform(patterns)

    print("Generated PDL_interp code:")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    demo_transformation()

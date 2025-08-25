"""
PDL to PDL_interp Transformation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from xdsl.context import Context
from xdsl.dialects import pdl
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser

# =============================================================================
# Core Data Structures - Positions
# =============================================================================


@dataclass(frozen=True)
class Position(ABC):
    """Base class for all position types"""

    parent: Optional["Position"] = None

    def get_operation_depth(self) -> int:
        """Returns depth of first ancestor operation position"""
        if isinstance(self, OperationPosition):
            return self.depth
        return self.parent.get_operation_depth() if self.parent else 0

    @abstractmethod
    def ranking(self) -> int:
        """Cost metric for ordering positions"""
        ...


@dataclass(frozen=True)
class OperationPosition(Position):
    """Represents an operation in the IR"""

    depth: int = 0

    def is_root(self) -> bool:
        return self.depth == 0

    def is_operand_defining_op(self) -> bool:
        return isinstance(self.parent, (OperandPosition | OperandGroupPosition))

    def ranking(self):
        return 0


@dataclass(frozen=True)
class OperandPosition(Position):
    """Represents an operand of an operation"""

    operand_number: int = -1

    def ranking(self):
        return 1


@dataclass(frozen=True)
class OperandGroupPosition(Position):
    """Represents a group of operands"""

    group_number: int | None = None
    is_variadic: bool = False

    def ranking(self):
        return 2


@dataclass(frozen=True)
class ResultPosition(Position):
    """Represents a result of an operation"""

    result_number: int = -1

    def ranking(self):
        return 3


# TODO: ResultGroupPosition?


@dataclass(frozen=True)
class AttributePosition(Position):
    """Represents an attribute of an operation"""

    attribute_name: str = ""

    def ranking(self):
        return 4


@dataclass(frozen=True)
class TypePosition(Position):
    """Represents the type of a value"""

    def ranking(self):
        return 5


@dataclass(frozen=True)
class UsersPosition(Position):
    """Represents users of a value"""

    use_representative: bool = False

    def ranking(self):
        return 6


# =============================================================================
# Predicate System - Questions and Answers
# =============================================================================


@dataclass(frozen=True)
class Predicate(ABC):
    """Base predicate class"""

    @abstractmethod
    def ranking(self) -> int:
        """Cost metric for ordering Predicates"""
        ...


@dataclass(frozen=True)
class Question(Predicate):
    """Represents a question/check to perform"""

    pass


@dataclass(frozen=True)
class Answer(Predicate):
    """Represents an expected answer to a question"""

    value: Any = None

    def ranking(self) -> int:
        """Ranking for ordering Answers"""
        return 0


# Question Types
@dataclass(frozen=True)
class IsNotNullQuestion(Question):
    def ranking(self) -> int:
        """Ranking for ordering Questions"""
        return 1


@dataclass(frozen=True)
class OperationNameQuestion(Question):
    def ranking(self) -> int:
        """Ranking for ordering Questions"""
        return 2


@dataclass(frozen=True)
class OperandCountQuestion(Question):
    def ranking(self) -> int:
        """Ranking for ordering Questions"""
        return 3


@dataclass(frozen=True)
class ResultCountQuestion(Question):
    def ranking(self) -> int:
        """Ranking for ordering Questions"""
        return 4


@dataclass(frozen=True)
class EqualToQuestion(Question):
    other_position: Position

    def ranking(self) -> int:
        """Ranking for ordering Questions"""
        return 5


# Answer Types
@dataclass(frozen=True)
class TrueAnswer(Answer):
    def ranking(self) -> int:
        """Ranking for ordering Answers"""
        # TODO: should this be 6 or restart the count for Answers?
        return 6


@dataclass(frozen=True)
class UnsignedAnswer(Answer):
    value: int = 0

    def ranking(self) -> int:
        """Ranking for ordering Answers"""
        return 7


@dataclass(frozen=True)
class StringAnswer(Answer):
    value: str = ""

    def ranking(self) -> int:
        """Ranking for ordering Answers"""
        return 8


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
        self._position_cache: dict[tuple[Any, ...], Position] = {}

    def get_root(self) -> OperationPosition:
        """Get the root operation position"""
        key = ("root",)
        if key not in self._position_cache:
            self._position_cache[key] = OperationPosition(depth=0, parent=None)
        return cast(OperationPosition, self._position_cache[key])

    def get_operand_defining_op(self, operand_pos: Position) -> OperationPosition:
        """Get the operation that defines an operand"""
        key = ("defining_op", operand_pos)
        if key not in self._position_cache:
            depth = operand_pos.get_operation_depth() + 1
            self._position_cache[key] = OperationPosition(
                depth=depth, parent=operand_pos
            )
        return cast(OperationPosition, self._position_cache[key])

    def get_operand(
        self, op_pos: OperationPosition, operand_num: int
    ) -> OperandPosition:
        """Get an operand position"""
        key = ("operand", op_pos, operand_num)
        if key not in self._position_cache:
            self._position_cache[key] = OperandPosition(
                operand_number=operand_num, parent=op_pos
            )
        return cast(OperandPosition, self._position_cache[key])

    def get_result(self, op_pos: OperationPosition, result_num: int) -> ResultPosition:
        """Get a result position"""
        key = ("result", op_pos, result_num)
        if key not in self._position_cache:
            self._position_cache[key] = ResultPosition(
                result_number=result_num, parent=op_pos
            )
        return cast(ResultPosition, self._position_cache[key])

    def get_attribute(
        self, op_pos: OperationPosition, attr_name: str
    ) -> AttributePosition:
        """Get an attribute position"""
        key = ("attribute", op_pos, attr_name)
        if key not in self._position_cache:
            self._position_cache[key] = AttributePosition(
                attribute_name=attr_name, parent=op_pos
            )
        return cast(AttributePosition, self._position_cache[key])

    def get_type(self, pos: Position) -> TypePosition:
        """Get a type position"""
        key = ("type", pos)
        if key not in self._position_cache:
            self._position_cache[key] = TypePosition(parent=pos)
        return cast(TypePosition, self._position_cache[key])

    # Predicate builders
    def get_is_not_null(self) -> tuple[Question, Answer]:
        return (IsNotNullQuestion(), TrueAnswer())

    def get_operation_name(self, name: str) -> tuple[Question, Answer]:
        return (OperationNameQuestion(), StringAnswer(value=name))

    def get_operand_count(self, count: int) -> tuple[Question, Answer]:
        return (OperandCountQuestion(), UnsignedAnswer(value=count))

    def get_result_count(self, count: int) -> tuple[Question, Answer]:
        return (ResultCountQuestion(), UnsignedAnswer(value=count))

    def get_equal_to(self, other_position: Position) -> tuple[Question, Answer]:
        return (EqualToQuestion(other_position=other_position), TrueAnswer())


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

    answer: Answer
    success_node: MatcherNode | None = None
    failure_node: MatcherNode | None = None

    success_node: MatcherNode | None = None


@dataclass
class SwitchNode(MatcherNode):
    """Multi-way switch node"""

    children: dict[Answer, MatcherNode | None] = field(default_factory=lambda: {})


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

    def detect_roots(self, pattern: pdl.PatternOp) -> list[pdl.OperationOp]:
        """Detect root operations in a pattern"""
        used: set[pdl.OperationOp] = set()

        for operation_op in pattern.body.ops:
            if not isinstance(operation_op, pdl.OperationOp):
                continue
            for operand in operation_op.operand_values:
                result_op = operand.owner
                if isinstance(result_op, pdl.ResultOp | pdl.ResultsOp):
                    assert isinstance(
                        used_op := result_op.parent_.owner, pdl.OperationOp
                    )
                    used.add(used_op)

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        if rewriter.root is not None:
            assert isinstance(root := rewriter.root.owner, pdl.OperationOp)
            if root in used:
                used.remove(root)

        roots = [
            op
            for op in pattern.body.ops
            if isinstance(op, pdl.OperationOp) and op not in used
        ]
        return roots

    def extract_tree_predicates(
        self,
        value: Operation | SSAValue,
        position: Position,
        inputs: dict[Operation | SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates by walking the operation tree"""
        predicates: list[PositionalPredicate] = []

        # Avoid revisiting values
        if value in inputs:
            if inputs[value] != position:
                # Add equality constraint
                q, a = self.builder.get_equal_to(inputs[value])
                predicates.append(PositionalPredicate(position, q, a))
            return predicates

        inputs[value] = position

        # Handle different PDL value types
        if isinstance(value, Operation) and isinstance(position, OperationPosition):
            predicates.extend(
                self._extract_operation_predicates(value, position, inputs)
            )
        elif isinstance(value, SSAValue) and isinstance(position, OperandPosition):
            predicates.extend(self._extract_operand_predicates(value, position, inputs))

        return predicates

    def _extract_operation_predicates(
        self,
        op_value: Operation,
        op_pos: OperationPosition,
        inputs: dict[Operation | SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operation"""
        predicates: list[PositionalPredicate] = []

        # Access PDL operation properties
        if not op_pos.is_root():
            q, a = self.builder.get_is_not_null()
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Operation name check for pdl.operation
        if isinstance(op_value, pdl.OperationOp) and op_value.opName:
            op_name = op_value.opName.data
            q, a = self.builder.get_operation_name(op_name)
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Operand count check
        if isinstance(op_value, pdl.OperationOp):
            operand_count = len(op_value.operand_values)
            if operand_count > 0:
                q, a = self.builder.get_operand_count(operand_count)
                predicates.append(PositionalPredicate(op_pos, q, a))

        # Result count check
        if isinstance(op_value, pdl.OperationOp):
            # PDL operations typically have one result of OperationType
            result_count = 1
            q, a = self.builder.get_result_count(result_count)
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Extract operand predicates
        if isinstance(op_value, pdl.OperationOp):
            for i, operand in enumerate(op_value.operand_values):
                operand_pos = self.builder.get_operand(op_pos, i)
                predicates.extend(
                    self.extract_tree_predicates(operand, operand_pos, inputs)
                )

        return predicates

    def _extract_operand_predicates(
        self,
        operand_value: SSAValue,
        operand_pos: OperandPosition,
        inputs: dict[Operation | SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operand"""
        predicates: list[PositionalPredicate] = []

        # Not-null check for non-root operands
        if operand_pos.parent:
            q, a = self.builder.get_is_not_null()
            predicates.append(PositionalPredicate(operand_pos, q, a))

        # If operand has a defining operation, recurse
        # Check if operand has a defining operation
        if operand_value.owner:
            op_pos = self.builder.get_operand_defining_op(operand_pos)
            if isinstance(operand_value.owner, Operation):
                predicates.extend(
                    self.extract_tree_predicates(operand_value.owner, op_pos, inputs)
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
            -hash(self.position.ranking()),  # Position dependency
            -hash(self.question.ranking()),  # Predicate dependency
            -self.tie_breaker,  # Deterministic order
        ) > (
            other.primary_score,
            other.secondary_score,
            -other.position.get_operation_depth(),
            -hash(other.position.ranking()),
            -hash(other.question.ranking()),
            -other.tie_breaker,
        )


class PredicateTreeBuilder:
    """Builds optimized predicate matching trees"""

    def __init__(self):
        self.analyzer = PatternAnalyzer(PredicateBuilder())

    def build_predicate_tree(self, patterns: list[Any]) -> MatcherNode:
        """Build optimized matcher tree from multiple patterns"""

        # Extract predicates for all patterns
        all_pattern_predicates: list[tuple[Any, list[PositionalPredicate]]] = []
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
        if root_node is not None:
            root_node = self._optimize_tree(root_node)
            root_node = self._insert_exit_node(root_node)
            return root_node
        else:
            # Return a default exit node if no patterns were processed
            return ExitNode()

    def _extract_pattern_predicates(
        self, pattern: pdl.PatternOp
    ) -> list[PositionalPredicate]:
        """Extract all predicates for a single pattern"""
        roots = self.analyzer.detect_roots(pattern)

        # For simplicity, use first root (in real implementation,
        # would use optimal root selection)
        if len(roots) > 1:
            raise NotImplementedError("Multiple roots not yet supported")
        root = roots[0] if roots else None
        if not root:
            return []

        inputs: dict[Operation | SSAValue, Position] = {}
        root_pos = self.analyzer.builder.get_root()

        return self.analyzer.extract_tree_predicates(root, root_pos, inputs)

    def _create_ordered_predicates(
        self, all_pattern_predicates: list[tuple[Any, list[PositionalPredicate]]]
    ) -> dict[tuple[Position, Question], OrderedPredicate]:
        """Create ordered predicates with frequency analysis"""
        predicate_map: dict[tuple[Position, Question], OrderedPredicate] = {}
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
            pattern_predicates: list[OrderedPredicate] = []

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
        pattern_predicates: dict[tuple[Position, Question], PositionalPredicate],
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

    def _insert_exit_node(self, root: MatcherNode) -> MatcherNode:
        """Insert exit node at end of failure paths"""
        root.failure_node = ExitNode()
        return root

    def _optimize_tree(self, root: MatcherNode) -> MatcherNode:
        """Optimize the tree by collapsing single-child switches to bools"""
        if isinstance(root, SwitchNode) and len(root.children) == 1:
            # Convert switch to bool node
            answer, child = next(iter(root.children.items()))
            bool_node = BoolNode(
                position=root.position,
                question=root.question,
                success_node=child,
                failure_node=root.failure_node,
                answer=answer,
            )
            return bool_node

        # Recursively optimize children
        if isinstance(root, SwitchNode):
            for answer in root.children:
                child_node = root.children[answer]
                if child_node is not None:
                    root.children[answer] = self._optimize_tree(child_node)
        elif isinstance(root, BoolNode):
            if root.success_node is not None:
                root.success_node = self._optimize_tree(root.success_node)

        if root.failure_node is not None:
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

        # pdl_interp.FuncOp(
        #     name,
        #     # type,

        # )

        code = [f"pdl_interp.func @{name}(%arg0: !pdl.operation) {{"]

        # Generate body
        entry_block = self._generate_matcher_code(root, "entry")
        code.extend(f"  {line}" for line in entry_block)

        code.append("}")
        return "\n".join(code)

    def _generate_matcher_code(self, node: MatcherNode, block_name: str) -> list[str]:
        """Generate code for a matcher node"""
        if isinstance(node, ExitNode):
            return ["pdl_interp.finalize"]

        elif isinstance(node, SuccessNode):
            code: list[str] = []
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
        code: list[str] = []

        # Get value at position
        if node.position is None:
            return ["// Error: node position is None"]

        value_access = self._generate_value_access(node.position)
        code.extend(value_access["setup"])
        current_value = value_access["result"]

        # Generate predicate check
        success_block = f"^bb{self.block_counter}"
        self.block_counter += 1
        failure_block = f"^bb{self.block_counter}"
        self.block_counter += 1

        if node.question is None:
            code.append("// Error: node question is None")
        else:
            predicate_code = self._generate_predicate_check(
                node.question, None, current_value, success_block, failure_block
            )
            code.extend(predicate_code)

        # Success block
        code.append(f"{success_block}:")
        if node.success_node is not None:
            success_code = self._generate_matcher_code(node.success_node, success_block)
            code.extend(f"  {line}" for line in success_code)

        # Failure block
        code.append(f"{failure_block}:")
        if node.failure_node is not None:
            failure_code = self._generate_matcher_code(node.failure_node, failure_block)
            code.extend(f"  {line}" for line in failure_code)

        return code

    def _generate_switch_node_code(self, node: SwitchNode) -> list[str]:
        """Generate code for switch node"""
        code: list[str] = []

        # Get value at position
        if node.position is None:
            return ["// Error: node position is None"]

        value_access = self._generate_value_access(node.position)
        code.extend(value_access["setup"])
        current_value = value_access["result"]

        # Generate switch operation
        default_block = f"^bb{self.block_counter}"
        self.block_counter += 1

        case_blocks: list[str] = []
        case_values: list[str] = []

        for answer, child_node in node.children.items():
            case_block = f"^bb{self.block_counter}"
            self.block_counter += 1
            case_blocks.append(case_block)
            case_values.append(self._answer_to_string(answer))

        # Generate switch instruction
        if node.question is not None:
            switch_op = self._generate_switch_operation(
                node.question, current_value, case_values, default_block, case_blocks
            )
        else:
            switch_op = ["// Error: node question is None"]
        code.extend(switch_op)

        # Generate case blocks
        for i, (_, child_node) in enumerate(node.children.items()):
            if i < len(case_blocks):
                case_block = case_blocks[i]
                code.append(f"{case_block}:")
                if child_node is not None:
                    case_code = self._generate_matcher_code(child_node, case_block)
                    code.extend(f"  {line}" for line in case_code)

        # Default block
        code.append(f"{default_block}:")
        if node.failure_node is not None:
            default_code = self._generate_matcher_code(node.failure_node, default_block)
            code.extend(f"  {line}" for line in default_code)

        return code

    def _generate_value_access(self, position: Position) -> dict[str, Any]:
        """Generate code to access value at position"""
        setup_code: list[str] = []

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
        answer: Answer | None,
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
            op_name = answer.value if answer and hasattr(answer, "value") else "unknown"
            return [
                f'pdl_interp.check_operation_name of {value} is "{op_name}" -> {success_block}, {failure_block}'
            ]

        elif isinstance(question, OperandCountQuestion):
            count = (
                answer.value
                if answer and hasattr(answer, "value") and answer.value is not None
                else 0
            )
            return [
                f"pdl_interp.check_operand_count of {value} is {count} -> {success_block}, {failure_block}"
            ]

        elif isinstance(question, ResultCountQuestion):
            count = (
                answer.value
                if answer and hasattr(answer, "value") and answer.value is not None
                else 0
            )
            return [
                f"pdl_interp.check_result_count of {value} is {count} -> {success_block}, {failure_block}"
            ]

        elif isinstance(question, EqualToQuestion):
            if hasattr(question, "other_position"):
                other_access = self._generate_value_access(question.other_position)
                return other_access["setup"] + [
                    f"pdl_interp.are_equal {value}, {other_access['result']} -> {success_block}, {failure_block}"
                ]
            else:
                return ["// Error: other_position is None for equal check"]

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

    def _answer_to_string(self, answer: Any) -> str:
        """Convert answer to string representation"""
        if hasattr(answer, "value") and answer.value is not None:
            return str(answer.value)
        elif isinstance(answer, str):
            return answer
        else:
            return str(answer) if answer is not None else "null"


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
    """Demonstrate the transformation with a comprehensive example"""

    print("=== PDL to PDL_interp Transformation Demo ===\n")

    # Mock PDL pattern that mimics real PDL structure
    class MockPDLPattern:
        def __init__(self, name: str, operations: list["MockOperation"]):
            self.name = name
            self.operations = operations
            self.body = MockRegion(operations)

        def __str__(self):
            return f"PDL Pattern: {self.name}"

    class MockRegion:
        def __init__(self, operations: list["MockOperation"]):
            self.block = MockBlock(operations)

    class MockBlock:
        def __init__(self, operations: list["MockOperation"]):
            self.ops = operations

    class MockOperation:
        def __init__(self, op_name: str, operand_count: int = 1, result_count: int = 1):
            self.name = op_name
            self._operand_count = operand_count
            self._result_count = result_count
            self.opName = MockStringAttr(op_name)
            self.operand_values = [f"operand_{i}" for i in range(operand_count)]
            self.results = [MockResult(self)]

        def get_operation_name(self) -> str:
            return self.name

        def get_operand_count(self) -> int:
            return self._operand_count

        def get_result_count(self) -> int:
            return self._result_count

        def get_operands(self) -> list[str]:
            return self.operand_values

        def __str__(self):
            return f"{self.name}({', '.join(self.operand_values)})"

    class MockStringAttr:
        def __init__(self, value: str):
            self.data = value

    class MockResult:
        def __init__(self, owner: MockOperation):
            self.owner = owner
            self.uses = []

    # Create realistic test patterns
    patterns = [
        # Pattern 1: Binary arithmetic operation (multiply)
        MockPDLPattern(
            "arith_mul_pattern",
            [MockOperation("arith.mulf", operand_count=2, result_count=1)],
        ),
        # Pattern 2: Unary operation (absolute value)
        MockPDLPattern(
            "math_abs_pattern",
            [MockOperation("math.absf", operand_count=1, result_count=1)],
        ),
        # Pattern 3: Chained operations
        MockPDLPattern(
            "complex_pattern",
            [
                MockOperation("arith.addf", operand_count=2, result_count=1),
                MockOperation("math.sqrt", operand_count=1, result_count=1),
            ],
        ),
    ]

    print(f"Created {len(patterns)} test patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"  {i}. {pattern}")
        for op in pattern.operations:
            print(f"     - {op}")

    print("\n" + "=" * 60)

    try:
        # Create transformer
        transformer = PDLToPDLInterpTransformer()

        # Transform patterns
        print("Starting transformation...")
        result = transformer.transform(patterns)

        print("\n" + "=" * 60)
        print("Generated PDL_interp matcher code:")
        print("-" * 40)
        print(result)
        print("-" * 40)

        # Analyze the result
        lines = result.split("\n")
        print("\nCode analysis:")
        print(f"  - Generated {len(lines)} lines of code")
        print(f"  - Contains {result.count('pdl_interp.')} PDL_interp operations")
        print(f"  - Has {result.count('^bb')} basic blocks")

        print("\nTransformation completed successfully! ✓")

    except Exception as e:
        print(f"\nTransformation failed with error: {e}")
        print("This is expected as the implementation is still a framework.")
        print("In a real implementation, this would generate working PDL_interp IR.")

    print("\n" + "=" * 60)
    print("Demo completed. This shows the structure of a PDL->PDL_interp transformer.")


def run_simple_test():
    """Run a minimal test to verify basic functionality"""
    print("\n=== Simple Functionality Test ===")

    try:
        # Test basic position creation
        builder = PredicateBuilder()
        root_pos = builder.get_root()
        _operand_pos = builder.get_operand(root_pos, 0)

        print("✓ Created positions")

        # Test predicate creation
        q, a = builder.get_operation_name("test.op")
        print(f"✓ Created predicate: {q.kind} -> {a.kind}")

        print("✓ Basic functionality test passed!")

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")


if __name__ == "__main__":
    pattern = """pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %one = pdl.attribute = 1 : i32
  %constop = pdl.operation "arith.constant" {"value" = %one} -> (%type : !pdl.type)
  %const = pdl.result 0 of %constop
  %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %mulop {
    pdl.replace %mulop with (%x : !pdl.value)
  }
}"""
    ctx = Context()
    ctx.load_dialect(pdl.PDL)
    # ctx.load_dialect(Test)

    parser = Parser(ctx, pattern)
    pattern = parser.parse_op()
    print(pattern)

    assert isinstance(pattern, pdl.PatternOp)

    x = PredicateTreeBuilder()._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]

    print(x)

    y = PredicateTreeBuilder().build_predicate_tree([pattern])
    print(y)

    code = PDLInterpCodeGenerator().generate_matcher_function(y)

    _ = None
    # demo_transformation()
    # run_simple_test()

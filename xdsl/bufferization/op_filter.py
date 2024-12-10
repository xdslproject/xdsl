from collections.abc import Callable
from enum import IntEnum
from typing import NamedTuple

from xdsl.ir import Dialect, Operation


class FilterType(IntEnum):
    """Filter type: A filter can either be a DENY filter or an ALLOW filter."""

    DENY = 0
    ALLOW = 1


class OpFilter:
    """
    An op filter entry. Filters can be used to specify which ops should be processed by
    the bufferization.
    """

    class Entry(NamedTuple):
        fn: Callable[[type[Operation]], bool]
        """If the filter function evaluates to `true`, the filter matches."""
        filter_type: FilterType

    _entries: list[Entry]
    """
    A list of filter entries that determine whether an op should be allowed or denied.
    If the filter has an ALLOW rule, only ops that are allowed and not denied are
    allowed. If the filter does not have an ALLOW rule, only ops that are not denied are
    allowed.
    """

    def __init__(self):
        self._entries = []

    def has_allow_rule(self) -> bool:
        """Return `true` if the filter has at least one ALLOW rule."""
        return FilterType.ALLOW in (e.filter_type for e in self._entries)

    def is_op_allowed(self, op: type[Operation]) -> bool:
        """
        Return whether the op is allowed or not.

        If the filter does not have an ALLOW rule, ops are allowed by default,
        unless they are explicitly marked as DENY. If the filter has at least one
        ALLOW rule, ops are denied by default and only allowed if they match
        an ALLOW rule and no DENY rule.
        """
        has_allow = self.has_allow_rule()
        matches_allow = False

        for entry in self._entries:
            if entry.fn(op):
                if entry.filter_type == FilterType.DENY:
                    return False
                if entry.filter_type == FilterType.ALLOW:
                    matches_allow = True

        return matches_allow if has_allow else True

    def allow_operation_filter(self, fn: Callable[[type[Operation]], bool]):
        """
        Allow the given operation or operation filter.

        This function adds an ALLOW entry.
        """
        self._entries.append(self.Entry(fn, FilterType.ALLOW))

    def allow_operation(self, op_t: type[Operation]):
        """
        Allow the given operation or operation filter.

        This function adds an ALLOW entry.
        """
        self.allow_operation_filter(lambda _op_t: _op_t is op_t)

    def allow_operations(self, op_ts: set[type[Operation]]):
        """
        Allow the given ops.

        This function adds one or multiple ALLOW entries.
        """
        self.allow_operation_filter(lambda op_t: op_t in op_ts)

    def deny_operation_filter(self, fn: Callable[[type[Operation]], bool]):
        """
        Deny the given operation filter.

        This function adds a DENY entry.
        """
        self._entries.append(self.Entry(fn, FilterType.DENY))

    def deny_operation(self, op_t: type[Operation]):
        """
        Deny the given operation.

        This function adds a DENY entry.
        """
        self.deny_operation_filter(lambda _op_t: _op_t is op_t)

    def deny_operations(self, op_ts: set[type[Operation]]):
        """
        Deny the given ops.

        This function adds one or multiple DENY entries.
        """
        self.deny_operation_filter(lambda op_t: op_t in op_ts)

    def allow_dialect(self, dialect: Dialect):
        """
        Allow the given dialect.

        This function adds one ALLOW entry.
        """
        self.allow_operations(set(dialect.operations))

    def deny_dialect(self, dialect: Dialect):
        """
        Deny the given dialect.

        This function adds one DENY entry.
        """
        self.deny_operations(set(dialect.operations))

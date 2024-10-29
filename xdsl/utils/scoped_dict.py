from __future__ import annotations

from typing import Generic, TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class ScopedDict(Generic[_Key, _Value]):
    """
    Class holding the Python values associated with SSAValues during an
    interpretation context. An environment is a stack of scopes, values are
    assigned to the current scope, but can be fetched from a parent scope.
    """

    _local_scope: dict[_Key, _Value]
    parent: ScopedDict[_Key, _Value] | None
    name: str | None

    def __init__(
        self,
        parent: ScopedDict[_Key, _Value] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._local_scope = {}
        self.parent = parent
        self.name = name

    def __getitem__(self, key: _Key) -> _Value:
        """
        Fetch key from environment. Attempts to first fetch from current scope,
        then from parent scopes. Raises KeyError error if not found.
        """
        local = self._local_scope.get(key)
        if local is not None:
            return local
        if self.parent is None:
            raise KeyError()
        return self.parent[key]

    def __setitem__(self, key: _Key, value: _Value):
        """
        Assign key to current scope. Raises InterpretationError if key already
        assigned to.
        """
        if key in self._local_scope:
            raise ValueError(
                f"Cannot overwrite value {self._local_scope[key]} for key {key}"
            )
        self._local_scope[key] = value

    def __contains__(self, key: _Key) -> bool:
        return (
            key in self._local_scope or self.parent is not None and key in self.parent
        )

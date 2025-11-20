from collections.abc import Iterable

from xdsl.dialect_interfaces import DialectInterface


class OpAsmDialectInterface(DialectInterface):
    """
    The OpAsm interface generally deals with making printed ir look less verbose.
    It implements two main functionalities - Aliases and Resources.

    ## Aliases
    If you have some commonly used types and attributes `AsmPrinter` can generate aliases for them through this interface.
    For example, `!my_dialect.type<a=3,b=4,c=5,d=tuple,e=another_type>` and `#my_dialect.attr<a=3>`
    can be aliased to `!my_dialect_type` and `#my_dialect_attr`, simplifying further references.

    **This is not yet implemented in xdsl**

    ## Resources
    Keep a dialect specific storage of blobs in the Dialect object.

    This functionality is usefull when we want to reference some objects
    while keeping them outside of ir.
    For example if we have a big dense array we might decide to store it in
    dialect's storage and use a key to reference it inside the ir.
    Or if we want some data to be shared across attributes we might wanna
    store it in the storage and use the same key to reference it
    in different places in the ir.

    This can help keep the ir clean and also give some performance improvements
    compared to keeping these resources tied to ir objects.
    """

    _blob_storage: dict[str, str] = {}

    def declare_resource(self, key: str) -> str:
        """
        Declare a resource in the storage.
        Does key deduplication, returns the key that is actually used in the storage.
        """
        # This deduplication is mainly needed when we create resources
        # programmatically and derive keys from value types.
        # In case of parsing we think that all equal keys point to the same resource.
        if key in self._blob_storage:
            counter = 0
            while key + f"_{counter}" in self._blob_storage:
                counter += 1
            key = key + f"_{counter}"

        self._blob_storage[key] = ""

        return key

    def parse_resource(self, key: str, val: str):
        """
        Check that val is a blob and update the key value with it.
        """
        if not val.startswith("0x"):
            raise ValueError(f"Blob must be a hex string, got: {val}")

        if key not in self._blob_storage:
            raise KeyError(f"Resource with key {key} wasn't declared")

        self._blob_storage[key] = val

    def lookup(self, key: str) -> str | None:
        """
        Get a value tied to a key.
        """
        return self._blob_storage.get(key)

    def build_resources(self, keys: Iterable[str]) -> dict[str, str]:
        """
        Return dict of resources for a provided list of keys.
        Filter out keys that haven't been assigned a value.
        Usually used for printing, when provided with list of
        keys referenced in the ir.
        """
        return {
            key: self._blob_storage[key] for key in keys if self._blob_storage.get(key)
        }

class DialectInterface:
    """
    A base class for dialects' interfaces.
    They usually define functionality which is dialect specific to some transformation.

    For example DialectInlinerInterface defines which dialect operations can be inlined and how.
    Dialects will implement this interface and the inlining transformation will query them through the base interface.

    The design logic tries to follow MLIR's dialect interfaces closely
    https://mlir.llvm.org/docs/Interfaces/#dialect-interfaces
    """

    pass


class OpAsmDialectInterface(DialectInterface):
    _blob_storage: dict[str, str] = {}

    ## Aliases - yet to be implemented

    ## Resources
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
        if not val.startswith("0x"):
            raise ValueError(f"Blob must be a hex string, got: {val}")

        if key not in self._blob_storage:
            raise KeyError(f"Resource with key {key} wasn't declared")

        self._blob_storage[key] = val

    def lookup(self, key: str) -> str | None:
        return self._blob_storage.get(key)

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

    # Keep a dialect specific storage of blobs in the Dialect object.

    # This functionality is usefull when we want to reference some objects
    # while keeping them outside of ir.
    # For example if we have a big dense array we might decide to store it in
    # dialect's storage and use a key to reference it inside the ir.
    # Or if we want some data to be shared across attributes we might wanna
    # store it in the storage and use the same key to reference it
    # in different places in the ir.

    # This can help keep the ir clean and also give some performance improvements
    # compared to keeping these resources tied to ir objects.

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

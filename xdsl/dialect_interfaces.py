class DialectInterface:
    pass


class OpAsmDialectInterface(DialectInterface):
    blob_storage: dict[str, str] = {}

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
        if key in self.blob_storage:
            counter = 0
            while key + f"_{counter}" in self.blob_storage:
                counter += 1
            key = key + f"_{counter}"

        self.blob_storage[key] = ""

        return key

    def parse_resource(self, key: str, val: str):
        if not val.startswith("0x"):
            raise ValueError(f"Blob must be a hex string, got: {val}")

        if key not in self.blob_storage:
            raise KeyError(f"Resource with key {key} wasn't declared")

        self.blob_storage[key] = val

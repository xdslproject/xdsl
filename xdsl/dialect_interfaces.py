class DialectInterface:
    pass


class OpAsmDialectInterface(DialectInterface):
    blob_storage: dict[str, str] = {}

    ## Aliases - yet to be implemented
    ## Resources
    def parse_resource(self, key: str, val: str) -> str:
        # deduplicate the key
        if key in self.blob_storage:
            counter = 0
            while key + f"_{counter}" in self.blob_storage:
                counter += 1
            key = key + f"_{counter}"

        if not val.startswith("0x"):
            raise ValueError(f"Blob must be a hex string, got: {val}")

        self.blob_storage[key] = val

        return key

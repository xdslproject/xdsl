def encode_mlir_escape(input: str):
    bytes_contents = bytearray()
    iter_string = iter(input)
    for c0 in iter_string:
        if c0 != "\\":
            bytes_contents += c0.encode()
        else:
            c1 = next(iter_string, None)
            if c1 is None:
                raise ValueError(rf"Invalid escape sequence \{c1}f!")
            match c1:
                case "n":
                    bytes_contents += b"\n"
                case "t":
                    bytes_contents += b"\t"
                case "\\":
                    bytes_contents += b"\\"
                case '"':
                    bytes_contents += b'"'
                case _:
                    c2 = next(iter_string, None)
                    if c2 is None:
                        raise ValueError(rf"Invalid escape sequence \{c0}{c1}f!")
                    bytes_contents += int(c1 + c2, 16).to_bytes(1)

    return bytes(bytes_contents)


def decode_mlir_escape(input: bytes):
    string = ""
    for byte in input:
        match byte:
            case 0x5C:  # ord("\\")
                string += "\\\\"
            case _ if 0x20 > byte or byte > 0x7E or byte == 0x22:
                string += f"\\{byte:02X}"
            case _:
                string += chr(byte)
    return string

def encode_mlir_escape(input: str):
    """
    Encodes a string using MLIR escape rules.
    """
    bytes_contents = bytearray()
    i = 0
    # Find the first escape sequence
    ei = input.find("\\", i)
    # No escape sequence found: use brrr encoder on the full thing and bail out
    if ei == -1:
        return input.encode()
    # Otherwise, iterate over the chunks
    while i < len(input):
        # Use brrr encoder on the rest of the input if no more escape sequences are found
        if ei == -1:
            bytes_contents += input[i:].encode()
            break
        # Otherwise, encode the chunk before the escape sequence
        bytes_contents += input[i:ei].encode()
        # Then handle the escape sequence
        i = ei
        i += 1
        c0 = input[i]
        match c0:
            case "n":
                bytes_contents += b"\n"
            case "t":
                bytes_contents += b"\t"
            case "\\":
                bytes_contents += b"\\"
            case '"':
                bytes_contents += b'"'
            case _:
                i += 1
                c1 = input[i]
                bytes_contents += int(c0 + c1, 16).to_bytes(1)
        i += 1
        # Find the next escape sequence, start over
        ei = input.find("\\", i)
    # Return the bytes, encoded with ❤️
    return bytes(bytes_contents)


def decode_mlir_escape(input: bytes):
    """
    Decode a byte string with MLIR escape sequences.
    MLIR allows parsing string literals escaping binary data, more general than Unicode.

    """
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

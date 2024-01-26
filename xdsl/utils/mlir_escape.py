def encode_mlir_escape(input: str):
    bytes_contents = bytearray()
    i = 0
    ei = input.find("\\", i)
    if ei == -1:
        return input.encode()
    while i < len(input):
        if ei == -1:
            bytes_contents += input[i:].encode()
            break
        bytes_contents += input[i:ei].encode()
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
        ei = input.find("\\", i)

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

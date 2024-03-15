from xdsl.dialects import x86


def test_register():
    rax = x86.register.RAX
    assert rax.is_allocated
    assert rax.register_name == "rax"

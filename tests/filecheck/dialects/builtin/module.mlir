// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
    builtin.module {
    }
    builtin.module attributes {"a" = "foo", "b" = "bar", "unit"} {
    }
}

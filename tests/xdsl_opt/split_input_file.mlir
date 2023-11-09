builtin.module {
}

// -----
builtin.module {
  "test.op"() : () -> ()
}

// -----
builtin.module {
  %x = "test.op"() : () -> i1
}

// -----
builtin.module {
  %x = "test.op"() : () -> i2
}

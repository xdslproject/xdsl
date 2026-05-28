// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

"builtin.module"() ({

  "test.op"() {name = "\""} : () -> ()
  // CHECK:      "test.op"() {name = "\""} : () -> ()

  "test.op"() {name = "\n"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "\n"} : () -> ()

  "test.op"() {name = "\t"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "\t"} : () -> ()

  "test.op"() {name = "\\"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "\\"} : () -> ()

  "test.op"() {name = "\E2\9A\A0\EF\B8\8FP\EDU\BA\01\00\10\00\A0\11\00\00\00\00\00\00\02\00\01\01@\00\00\00(\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00=\00\00\00\00\00\00\00\00\00\00\00\11\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00x\00\00\00\00\00\00\00\00\00\00\00\80\0C\00\00\00\00\00\00\80\0A\00\00\00\00\00\00=\05=\00@\008\00\03\00@\00\08\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00.text.gpu_kernel_kernel"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "\E2\9A\A0\EF\B8\8FP\EDU\BA\01\00\10\00\A0\11\00\00\00\00\00\00\02\00\01\01@\00\00\00(\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\07\00\01\00=\00\00\00\00\00\00\00\00\00\00\00\11\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00x\00\00\00\00\00\00\00\00\00\00\00\80\0C\00\00\00\00\00\00\80\0A\00\00\00\00\00\00=\05=\00@\008\00\03\00@\00\08\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00.text.gpu_kernel_kernel"} : () -> ()

  "test.op"() {name = "\22quoted_attr\22"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "\"quoted_attr\""} : () -> ()

  "test.op"() {name = "\202"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = " 2"} : () -> ()

  // ASCII-only string with control bytes — exercises the StringAttr print path.
  // Pre-fix this round-tripped through json.dumps producing "\uXXXX" escapes
  // that xDSL's (and mlir-opt's) lexer reject.
  "test.op"() {name = "NUL:\00 BEL:\07 ESC:\1B DEL:\7F end"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "NUL:\00 BEL:\07 ESC:\1B DEL:\7F end"} : () -> ()

  "test.op"() {name = "Hello\00\00"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "Hello\00\00"} : () -> ()

  "test.op"() {name = "CR\0Dhere"} : () -> ()
  // CHECK-NEXT: "test.op"() {name = "CR\0Dhere"} : () -> ()

}) : () -> ()

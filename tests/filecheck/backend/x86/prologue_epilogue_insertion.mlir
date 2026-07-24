// RUN: xdsl-opt --split-input-file -p "x86-prologue-epilogue-insertion" %s | filecheck %s

// CHECK: func @main
x86_func.func @main() {
  // CHECK-NEXT: %{{.*}} = x86.get_register : !x86.reg64<rsp>
  // CHECK-NEXT: %{{.*}} = x86.get_register : !x86.reg64<r14>
  // CHECK-NEXT: %{{.*}} = x86.s.push %{{.*}}, %{{.*}} : (!x86.reg64<rsp>, !x86.reg64<r14>) -> !x86.reg64<rsp>
  // CHECK-NEXT: %{{.*}} = x86.get_register : !x86.reg64<r15>
  // CHECK-NEXT: %{{.*}} = x86.s.push %{{.*}}, %{{.*}} : (!x86.reg64<rsp>, !x86.reg64<r15>) -> !x86.reg64<rsp>

  // CHECK-NEXT: %r12 = x86.get_register : !x86.reg64<r12>
  // CHECK-NEXT: %r13 = x86.get_register : !x86.reg64<r13>
  // CHECK-NEXT: %r14 = x86.rs.add %r12, %r13 : (!x86.reg64<r12>, !x86.reg64<r13>) -> !x86.reg64<r14>
  // CHECK-NEXT: %r8 = x86.get_register : !x86.reg64<r8>
  // CHECK-NEXT: %r15 = x86.ds.mov %r8 : (!x86.reg64<r8>) -> !x86.reg64<r15>
  // CHECK-NEXT: %rflags = x86.ss.cmp %r8, %r15 : (!x86.reg64<r8>, !x86.reg64<r15>) -> !x86.rflags<rflags>
  // CHECK-NEXT: x86.c.jne %rflags : !x86.rflags<rflags>, ^then(%r8 : !x86.reg64<r8>), ^else(%r15 : !x86.reg64<r15>)
  %r12 = x86.get_register : !x86.reg64<r12>
  %r13 = x86.get_register : !x86.reg64<r13>
  %r14 = x86.rs.add %r12,%r13: (!x86.reg64<r12>, !x86.reg64<r13>) -> !x86.reg64<r14>
  %r8 = x86.get_register : !x86.reg64<r8>
  %r15 = x86.ds.mov %r8 : (!x86.reg64<r8>) -> !x86.reg64<r15>
  %rflags = x86.ss.cmp %r8, %r15 : (!x86.reg64<r8>, !x86.reg64<r15>) -> !x86.rflags<rflags>
  x86.c.jne %rflags : !x86.rflags<rflags>, ^then(%r8 : !x86.reg64<r8>), ^else(%r15 : !x86.reg64<r15>)
^else:
  // CHECK-NEXT: ^else
  // CHECK-NEXT: x86.label "else"
  // CHECK-NEXT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<r15>)
  // CHECK-NEXT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<r14>)
  // CHECK-NEXT: x86_func.ret
  x86.label "else"
  x86_func.ret
^then:
  // CHECK-NEXT: ^then
  // CHECK-NEXT: x86.label "then"
  // CHECK-NEXT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<r15>)
  // CHECK-NEXT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg64<rsp>) -> (!x86.reg64<rsp>, !x86.reg64<r14>)
  // CHECK-NEXT: x86_func.ret
  x86.label "then"
  x86_func.ret
}

// -----

// CHECK: func @simple
x86_func.func @simple(%0: !x86.reg64<rdi>, %1: !x86.reg64<rsi>) -> !x86.reg64<rax> {
  // CHECK-NOT: %{{.*}} = x86.get_register : !x86.reg64<rsp>
  // CHECK-NOT: %{{.*}} = x86.s.push %{{.*}}, %{{.*}} : (!x86.reg64<rsp>, !x86.reg64<{{.*}}>) -> !x86.reg64<rsp>
  // CHECK-NEXT: %2 = x86.ds.mov %0 : (!x86.reg64<rdi>) -> !x86.reg64<r8>
  // CHECK-NEXT: %3 = x86.ds.mov %1 : (!x86.reg64<rsi>) -> !x86.reg64<r9>
  // CHECK-NEXT: %4 = x86.rs.add %2, %3 : (!x86.reg64<r8>, !x86.reg64<r9>) -> !x86.reg64<r10>
  // CHECK-NEXT: %5 = x86.ds.mov %4 : (!x86.reg64<r10>) -> !x86.reg64<rax>
  %2 = x86.ds.mov %0 : (!x86.reg64<rdi>) -> !x86.reg64<r8>
  %3 = x86.ds.mov %1 : (!x86.reg64<rsi>) -> !x86.reg64<r9>
  %4 = x86.rs.add %2,%3: (!x86.reg64<r8>, !x86.reg64<r9>) -> !x86.reg64<r10>
  %5 = x86.ds.mov %4 : (!x86.reg64<r10>) -> !x86.reg64<rax>
  // CHECK-NOT: %{{.*}}, %{{.*}} = x86.d.pop %{{.*}} : (!x86.reg64<rsp>) -> (!x86.reg64<{{.*}}>, !x86.reg64<rsp>)
  // CHECK-NEXT: x86_func.ret
  x86_func.ret
}

// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
    %0 = x86.get_register : () -> !x86.reg<>
    %1 = x86.get_register : () -> !x86.reg<>

    // add, sub, imul, idiv, not, and, or, xor, mov, push, pop 
    %add = x86.add %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK: %{{.*}} = x86.add %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %sub = x86.sub %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.sub %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %mul = x86.imul %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.imul %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %div = x86.idiv %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.idiv %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %not = x86.not %0 : (!x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.not %{{.*}} : (!x86.reg<>) -> !x86.reg<>
    %and = x86.and %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.and %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %or = x86.or %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.or %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %xor = x86.xor %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.xor %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    %mov = x86.mov %0, %1 : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.mov %{{.*}}, %{{.*}} : (!x86.reg<>, !x86.reg<>) -> !x86.reg<>
    x86.push %0 : (!x86.reg<>) -> ()
    // CHECK-NEXT: x86.push %{{.*}} : (!x86.reg<>)
    %pop = x86.pop  : () -> !x86.reg<>
    // CHECK-NEXT: %{{.*}} = x86.pop : () -> !x86.reg<>

    // vfmadd231pd, vmovapd, vbroadcastsd
    %2 = x86.get_avx_register : () -> !x86.avxreg<>
    %3 = x86.get_avx_register : () -> !x86.avxreg<>
    %4 = x86.get_avx_register : () -> !x86.avxreg<>

    %vfmadd231 = x86.vfmadd231pd %2, %3, %4 : (!x86.avxreg<>, !x86.avxreg<>, !x86.avxreg<>) -> !x86.avxreg<>
    // CHECK: %{{.*}} = x86.vfmadd231pd %{{.*}}, %{{.*}}, %{{.*}} : (!x86.avxreg<>, !x86.avxreg<>, !x86.avxreg<>) -> !x86.avxreg<>
}) : () -> ()
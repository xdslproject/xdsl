CC=clang -march=x86-64
LD=clang

check: exe
	@./$<

exe: matmul.o main.o matmul.o
	@$(LD) $^ -o $@

main.o: main.c
	@$(CC) -c $< -o $@

matmul.o: matmul.s
	@$(CC) -c $< -o $@ -Wa,--noexecstack

matmul.s: integration_test_matmul.mlir
	 @uv run xdsl-opt $< -p convert-vector-to-ptr,convert-memref-to-ptr{lower_func=true},convert-ptr-type-offsets,canonicalize,convert-func-to-x86-func,convert-vector-to-x86{arch=avx2},convert-ptr-to-x86{arch=avx2},convert-arith-to-x86,reconcile-unrealized-casts,canonicalize,x86-infer-broadcast,dce,x86-allocate-registers,canonicalize -t x86-asm > $@

clean:
	rm -f main.o matmul.o matmul.s exe

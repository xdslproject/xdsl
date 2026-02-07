// RUN: xdsl-opt -t python-source %s | filecheck %s

py.func @main() {
  // CHECK: _0 = 0
  %0 = py.const 0
  // CHECK: _1 = 1  
  %1 = py.const 1
  // CHECK: _2 = _0 + _1
  %2 = py.binop "add" %0 %1
  // CHECK: return _2
  py.return %2
}

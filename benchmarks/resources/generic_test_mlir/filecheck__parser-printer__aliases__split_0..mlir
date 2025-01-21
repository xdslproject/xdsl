"builtin.module"() ({
  "test.op"() {attr = 1 : i32} : () -> ()
  "test.op"() {attr = i32} : () -> ()
  "test.op"() {attr = vector<1xi32>} : () -> ()
  "test.op"() {attr = 0 : i32} : () -> ()
  "test.op"() {attr = [0 : i32]} : () -> ()
}) : () -> ()

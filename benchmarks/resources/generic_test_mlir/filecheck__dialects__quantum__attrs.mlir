"builtin.module"() ({
  "test.op"() {angle = !quantum.angle<0>} : () -> ()
  "test.op"() {angle = !quantum.angle<pi>} : () -> ()
  "test.op"() {angle = !quantum.angle<2pi>} : () -> ()
  "test.op"() {angle = !quantum.angle<pi:2>} : () -> ()
  "test.op"() {angle = !quantum.angle<3pi:2>} : () -> ()
  "test.op"() {angle = !quantum.angle<5pi:2>} : () -> ()
  "test.op"() {angle = !quantum.angle<-pi>} : () -> ()
}) : () -> ()

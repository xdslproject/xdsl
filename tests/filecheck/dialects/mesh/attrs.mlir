// RUN: XDSL_ROUNDTRIP

"test.op"() {attrs = [
            
            #mesh<axisarray[[]]>,
            // CHECK: #mesh<axisarray[[]]>

            #mesh<axisarray[[21]]>,
            // CHECK-SAME: #mesh<axisarray[[21]]>

            #mesh<axisarray[[15, 234]]>,
            // CHECK-SAME: #mesh<axisarray[[15, 234]]>

            #mesh<axisarray[[15, 234], [], [12]]>,
            // CHECK-SAME: #mesh<axisarray[[15, 234], [], [12]]>

            #mesh<axisarray[[15, 234], [56], [12]]>
            // CHECK-SAME: #mesh<axisarray[[15, 234], [56], [12]]>

        ]} : () -> ()

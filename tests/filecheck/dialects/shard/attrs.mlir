// RUN: XDSL_ROUNDTRIP

"test.op"() {attrs = [
            
            #shard<axisarray[[]]>,
            // CHECK: #shard<axisarray[[]]>

            #shard<axisarray[[21]]>,
            // CHECK-SAME: #shard<axisarray[[21]]>

            #shard<axisarray[[15, 234]]>,
            // CHECK-SAME: #shard<axisarray[[15, 234]]>

            #shard<axisarray[[15, 234], [], [12]]>,
            // CHECK-SAME: #shard<axisarray[[15, 234], [], [12]]>

            #shard<axisarray[[15, 234], [56], [12]]>
            // CHECK-SAME: #shard<axisarray[[15, 234], [56], [12]]>

        ]} : () -> ()

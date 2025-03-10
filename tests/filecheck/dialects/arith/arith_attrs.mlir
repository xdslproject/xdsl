// RUN: XDSL_ROUNDTRIP

"test.op"() {attrs = [
                #arith.fastmath<reassoc>,
                // CHECK: #arith.fastmath<reassoc>
                #arith<fastmath<reassoc>>,
                // CHECK-SAME: #arith.fastmath<reassoc>
                #arith.fastmath<nnan>,
                // CHECK-SAME: #arith.fastmath<nnan>
                #arith.fastmath<ninf>,
                // CHECK-SAME: #arith.fastmath<ninf>
                #arith.fastmath<nsz>,
                // CHECK-SAME: #arith.fastmath<nsz>
                #arith.fastmath<arcp>,
                // CHECK-SAME: #arith.fastmath<arcp>
                #arith.fastmath<contract>,
                // CHECK-SAME: #arith.fastmath<contract>
                #arith.fastmath<afn>,
                // CHECK-SAME: #arith.fastmath<afn>
                #arith.fastmath<none>,
                // CHECK-SAME: #arith.fastmath<none>
                #arith.fastmath<fast>,
                // CHECK-SAME: #arith.fastmath<fast>
                #arith.fastmath<nnan,nsz>
                // CHECK-SAME: #arith.fastmath<nnan,nsz>
            ]}: () -> ()

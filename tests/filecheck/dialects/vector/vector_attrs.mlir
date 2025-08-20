// RUN: XDSL_ROUNDTRIP

"test.op"() {
    attrs = [
        #vector.kind<add>,
        // CHECK: #vector.kind<add>
        #vector.kind<mul>,
        // CHECK: #vector.kind<mul>
        #vector.kind<minui>,
        // CHECK: #vector.kind<minui>
        #vector.kind<minsi>,
        // CHECK: #vector.kind<minsi>
        #vector.kind<minnumf>,
        // CHECK: #vector.kind<minnumf>
        #vector.kind<maxui>,
        // CHECK: #vector.kind<maxui>
        #vector.kind<maxsi>,
        // CHECK: #vector.kind<maxsi>
        #vector.kind<maxnumf>,
        // CHECK: #vector.kind<maxnumf>
        #vector.kind<and>,
        // CHECK: #vector.kind<and>
        #vector.kind<or>,
        // CHECK: #vector.kind<or>
        #vector.kind<xor>,
        // CHECK: #vector.kind<xor>
        #vector.kind<maximumf>,
        // CHECK: #vector.kind<maximumf>
        #vector.kind<minimumf>
        // CHECK: #vector.kind<minimumf>
    ]
}: () -> ()

# RUN: python %s | filecheck %s

from xdsl.dialects.irdl.pyrdl_to_irdl import dialect_to_irdl
from xdsl.dialects.cmath import CMath

print(dialect_to_irdl(CMath, "cmath"))

# CHECK:      irdl.dialect @cmath {

# CHECK-NEXT:   irdl.attribute @cmath.complex {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.parameters(%{{.*}})
# CHECK-NEXT:   }

# CHECK-NEXT:   irdl.operation @cmath.norm {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.operands(%{{.*}})
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.results(%{{.*}})
# CHECK-NEXT:   }

# CHECK-NEXT:   irdl.operation @cmath.mul {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.operands(%{{.*}}, %{{.*}})
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.results(%{{.*}})
# CHECK-NEXT:   }

# CHECK-NEXT: }

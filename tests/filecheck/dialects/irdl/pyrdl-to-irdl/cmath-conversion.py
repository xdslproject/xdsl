# RUN: python %s | filecheck %s

from xdsl.dialects.cmath import Cmath
from xdsl.dialects.irdl.pyrdl_to_irdl import dialect_to_irdl

print(dialect_to_irdl(Cmath, "cmath"))

# CHECK:      irdl.dialect @cmath {

# CHECK-NEXT:   irdl.attribute @complex {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.parameters(%{{.*}})
# CHECK-NEXT:   }

# CHECK-NEXT:   irdl.operation @norm {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.operands {
# CHECK-NEXT:       "in" = %{{.*}}
# CHECK-NEXT:     }
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.results {
# CHECK-NEXT:       "out" = %{{.*}}
# CHECK-NEXT:     }
# CHECK-NEXT:   }

# CHECK-NEXT:   irdl.operation @mul {
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.operands {
# CHECK-NEXT:       "lhs" = %{{.*}},
# CHECK-NEXT:       "rhs" = %{{.*}}
# CHECK-NEXT:     }
# CHECK-NEXT:     %{{.*}} = irdl.any
# CHECK-NEXT:     irdl.results {
# CHECK-NEXT:       "out" = %{{.*}}
# CHECK-NEXT:     }
# CHECK-NEXT:   }

# CHECK-NEXT: }

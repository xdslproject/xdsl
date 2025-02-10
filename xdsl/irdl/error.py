from enum import Enum


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1
    AttributeDefAnnot = 2
    OptAttributeDefAnnot = 3
    SingleBlockRegionAnnot = 4
    ConstraintVarAnnot = 5

from xdsl.ir import Dialect

from .attributes import LabelAttr
from .ops import (
    DirectiveOp,
    GetRegisterOp,
    LabelOp,
    M_IDivOp,
    M_ImulOp,
    M_NegOp,
    M_NotOp,
    M_PopOp,
    M_PushOp,
    MI_AddOp,
    MI_AndOp,
    MI_MovOp,
    MI_OrOp,
    MI_SubOp,
    MI_XorOp,
    MR_AddOp,
    MR_AndOp,
    MR_MovOp,
    MR_OrOp,
    MR_SubOp,
    MR_XorOp,
    R_IDivOp,
    R_ImulOp,
    R_NotOp,
    R_PopOp,
    R_PushOp,
    RI_AddOp,
    RI_AndOp,
    RI_MovOp,
    RI_OrOp,
    RI_SubOp,
    RI_XorOp,
    RM_AddOp,
    RM_AndOp,
    RM_CmpOp,
    RM_ImulOp,
    RM_MovOp,
    RM_OrOp,
    RM_SubOp,
    RM_XorOp,
    RMI_ImulOp,
    RR_AddOp,
    RR_AndOp,
    RR_CmpOp,
    RR_ImulOp,
    RR_MovOp,
    RR_OrOp,
    RR_SubOp,
    RR_XorOp,
    RRI_ImulOP,
    S_JaeOp,
    S_JaOp,
    S_JbeOp,
    S_JbOp,
    S_JcOp,
    S_JeOp,
    S_JgeOp,
    S_JgOp,
    S_JleOp,
    S_JlOp,
    S_JmpOp,
    S_JnaeOp,
    S_JnaOp,
    S_JnbeOp,
    S_JnbOp,
    S_JncOp,
    S_JneOp,
    S_JngeOp,
    S_JngOp,
    S_JnleOp,
    S_JnlOp,
    S_JnoOp,
    S_JnpOp,
    S_JnsOp,
    S_JnzOp,
    S_JoOp,
    S_JpeOp,
    S_JpoOp,
    S_JpOp,
    S_JsOp,
    S_JzOp,
)
from .register import GeneralRegisterType, RFLAGSRegisterType

X86 = Dialect(
    "x86",
    [
        RR_AddOp,
        RR_SubOp,
        RR_ImulOp,
        RR_AndOp,
        RR_OrOp,
        RR_XorOp,
        RR_MovOp,
        RR_CmpOp,
        R_PushOp,
        R_PopOp,
        R_NotOp,
        R_IDivOp,
        R_ImulOp,
        RM_AddOp,
        RM_SubOp,
        RM_ImulOp,
        RM_AndOp,
        RM_OrOp,
        RM_XorOp,
        RM_CmpOp,
        RM_MovOp,
        RI_AddOp,
        RI_SubOp,
        RI_AndOp,
        RI_OrOp,
        RI_XorOp,
        RI_MovOp,
        MR_AddOp,
        MR_SubOp,
        MR_AndOp,
        MR_OrOp,
        MR_XorOp,
        MR_MovOp,
        MI_AddOp,
        MI_SubOp,
        MI_AndOp,
        MI_OrOp,
        MI_XorOp,
        MI_MovOp,
        RRI_ImulOP,
        RMI_ImulOp,
        M_PushOp,
        M_PopOp,
        M_NegOp,
        M_NotOp,
        M_IDivOp,
        M_ImulOp,
        S_JmpOp,
        S_JaOp,
        S_JaeOp,
        S_JbOp,
        S_JbeOp,
        S_JcOp,
        S_JeOp,
        S_JgOp,
        S_JgeOp,
        S_JlOp,
        S_JleOp,
        S_JnaOp,
        S_JnaeOp,
        S_JnbOp,
        S_JnbeOp,
        S_JncOp,
        S_JneOp,
        S_JngOp,
        S_JngeOp,
        S_JnlOp,
        S_JnleOp,
        S_JnoOp,
        S_JnpOp,
        S_JnsOp,
        S_JnzOp,
        S_JoOp,
        S_JpOp,
        S_JpeOp,
        S_JpoOp,
        S_JsOp,
        S_JzOp,
        GetRegisterOp,
        LabelOp,
        DirectiveOp,
    ],
    [
        GeneralRegisterType,
        RFLAGSRegisterType,
        LabelAttr,
    ],
)

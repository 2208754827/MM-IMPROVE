# Ultralytics YOLOMM - Neck Module
# SOEP (Small Object Enhance Pyramid) 小目标增强金字塔模块集

from .auxiliary import GSConvE, MFM, SNI
from .soep import CSPOmniKernel, FGM, OmniKernel, SPDConv
from .cgrfpn import (
    PyramidContextExtraction,
    GetIndexOutput,
    RCM,
    FuseBlockMulti,
    DynamicInterpolationFusion,
)
from .contextguide import ContextGuideFusionModule
from .fdpn_dasi import DASI
from .fdpn import FocusFeature
from .pacapn import ParallelAtrousConv, CSP_PAC, AttentionUpsample, AttentionDownsample
from .hypercompute import HyperComputeModule, MANet, MANet_Star, MANet_FasterBlock, MANet_FasterCGLU
from .wfu import WFU
# Neck 变体（AFPN/HSFPN/CFPT/融合等）
from .neck_variants import (
    GSConv,
    VoVGSCSP,
    AFPN_P345,
    AFPN_P345_Custom,
    AFPN_P2345,
    AFPN_P2345_Custom,
    Zoom_cat,
    ScalSeq,
    DynamicScalSeq,
    HFP,
    SDP,
    SDP_Improved,
    ChannelAttention_HSFPN,
    ELA_HSFPN,
    CA_HSFPN,
    CAA_HSFPN,
    FSA,
    Add,
    Multiply,
    CrossLayerSpatialAttention,
    CrossLayerChannelAttention,
    CrossAttentionBlock,
    FreqFusion,
    LocalSimGuidedSampler,
    PSFM,
    Fusion,
    GDSAFusion,
    GLSA,
    CAFMFusion,
    SDI,
    CSPStage,
    BiFusion,
    OREPANCSPELAN4,
    SBA,
    DPCF,
    HyperACE,
    FullPAD_Tunnel,
    EUCB,
    EUCB_SC,
    MSDC,
    MSCB,
    CSP_MSCB,
    MSCB_SC,
    CSP_MSCB_SC,
)

__all__ = [
    # SOEP核心模块
    'SPDConv',
    'FGM',
    'OmniKernel',
    'CSPOmniKernel',
    # SOEP辅助模块
    'SNI',
    'GSConvE',
    'MFM',
    # CGRFPN blocks
    'PyramidContextExtraction',
    'GetIndexOutput',
    'RCM',
    'FuseBlockMulti',
    'DynamicInterpolationFusion',
    'ContextGuideFusionModule',
    'DASI',
    'FocusFeature',
    'ParallelAtrousConv',
    'CSP_PAC',
    'AttentionUpsample',
    'AttentionDownsample',
    'HyperComputeModule',
    'MANet',
    'MANet_Star',
    'MANet_FasterBlock',
    'MANet_FasterCGLU',
    'WFU',
    # 颈部变体
    'GSConv','VoVGSCSP',
    'AFPN_P345','AFPN_P345_Custom','AFPN_P2345','AFPN_P2345_Custom',
    'Zoom_cat','ScalSeq','DynamicScalSeq',
    'HFP','SDP','SDP_Improved',
    'ChannelAttention_HSFPN','ELA_HSFPN','CA_HSFPN','CAA_HSFPN','FSA','Add','Multiply',
    'CrossLayerSpatialAttention','CrossLayerChannelAttention','CrossAttentionBlock',
    'FreqFusion','LocalSimGuidedSampler','PSFM',
    'Fusion','GDSAFusion','GLSA','CAFMFusion','SDI','CSPStage','BiFusion','OREPANCSPELAN4',
    'SBA','DPCF','HyperACE','FullPAD_Tunnel','EUCB','EUCB_SC','MSDC','MSCB','MSCB_SC','CSP_MSCB','CSP_MSCB_SC',
]

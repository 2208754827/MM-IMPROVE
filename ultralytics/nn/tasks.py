# Ultralytics 馃殌 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
# Extra modules conditional import锛堥粯璁ょ鐢級
# from ultralytics.nn.extra_modules import *
# from ultralytics.nn.backbone.convnextv2 import *
# from ultralytics.nn.backbone.fasternet import *
# from ultralytics.nn.backbone.efficientViT import *
# from ultralytics.nn.backbone.EfficientFormerV2 import *
# from ultralytics.nn.backbone.VanillaNet import *
# from ultralytics.nn.backbone.revcol import *
# from ultralytics.nn.backbone.lsknet import *
# from ultralytics.nn.backbone.SwinTransformer import *
# from ultralytics.nn.backbone.repvit import *
# from ultralytics.nn.backbone.CSwomTramsformer import *
# from ultralytics.nn.backbone.UniRepLKNet import *
# from ultralytics.nn.backbone.TransNext import *
# from ultralytics.nn.backbone.rmt import *
# from ultralytics.nn.backbone.pkinet import *
# from ultralytics.nn.backbone.mobilenetv4 import *
# from ultralytics.nn.backbone.starnet import *
# from ultralytics.nn.backbone.inceptionnext import *
# from ultralytics.nn.extra_modules.mobileMamba.mobilemamba import *
# from ultralytics.nn.backbone.MambaOut import *
# from ultralytics.nn.backbone.overlock import *
# from ultralytics.nn.backbone.lsnet import *
# except:
#     pass
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.backbone import (
    TimmBackbone,
    convnextv2_atto,
    convnextv2_base,
    convnextv2_femto,
    convnextv2_huge,
    convnextv2_large,
    convnextv2_nano,
    convnextv2_pico,
    convnextv2_tiny,
    repvit_m0_9,
    repvit_m1_0,
    repvit_m1_1,
    repvit_m1_5,
    repvit_m2_3,
    efficientformerv2_s0,
    efficientformerv2_s1,
    efficientformerv2_s2,
    efficientformerv2_l,
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
    SwinTransformer_Tiny,
)
from ultralytics.nn.modules import (
    # non-fusion modules (keep explicit import)
    AIFI,
    AIFI_DyT,
    AIFI_EDFFN,
    AIFI_LPE,
    AIFI_Mona,
    AIFI_RepBN,
    AIFI_SEFFN,
    AIFI_SEFN,
    TransformerEncoderLayer_AdditiveTokenMixer,
    TransformerEncoderLayer_ASSA,
    TransformerEncoderLayer_ASSA_SEFN,
    TransformerEncoderLayer_ASSA_SEFN_Mona,
    TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
    TransformerEncoderLayer_DAttention,
    TransformerEncoderLayer_DHSA,
    TransformerEncoderLayer_DPB,
    TransformerEncoderLayer_EfficientAdditiveAttnetion,
    TransformerEncoderLayer_HiLo,
    TransformerEncoderLayer_LocalWindowAttention,
    TransformerEncoderLayer_MSLA,
    TransformerEncoderLayer_MSMHSA,
    TransformerEncoderLayer_Pola,
    TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
    TransformerEncoderLayer_Pola_SEFN,
    TransformerEncoderLayer_Pola_SEFN_Mona,
    TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
    TransformerEncoderLayer_TSSA,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    BasicBlock,
    Blocks,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvNormLayer,
    ConvTranspose,
    FourierConv,
    Detect,
    v8Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    MCFGatedFusion,
    # YOLO-RD modules (kept explicit only for non-fusion entries)
    LRPCHead,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RGCSPELAN,
    RepVGGDW,
    ResNetLayer,
    RTDETRBottleNeck,
    RTDETRDecoder,
    SCDown,
    Segment,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
    get_activation,
)

# Import all fusion modules from the dedicated fusion package to avoid missing symbols
# and keep tasks.py decoupled from individual fusion module names.
from ultralytics.nn.modules.fusion import *  # noqa: F401,F403
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)

# 澶氭ā鎬佺粍浠跺鍏?(浠呭鍏ュ凡杩佺Щ鐨勭粍浠?
try:
    from ultralytics.nn.mm import MultiModalRouter, MultiModalConfigParser, HookManager
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
try:
    # Contrast components are optional and only used if hooks are present
    from ultralytics.nn.mm.contrast import ContrastController, ContrastConfig
    CONTRAST_AVAILABLE = True
except Exception:
    CONTRAST_AVAILABLE = False

_LSCD_IMPORT_ERROR = None
# LSCD 妫€娴嬪ご瀵煎叆 (杞婚噺鍖栧叡浜嵎绉娴嬪ご)
try:
    from ultralytics.nn.Head import (
        Detect_LSCD,
        Segment_LSCD,
        Pose_LSCD,
        OBB_LSCD,
        Conv_GN,
        Scale,
    )
    LSCD_AVAILABLE = True
except ImportError as _err:
    _LSCD_IMPORT_ERROR = _err
    LSCD_AVAILABLE = False
    LOGGER.warning(f"LSCD head import failed: {_err}")
    def _missing_lscd(*args, **kwargs):
        raise RuntimeError(f"LSCD head modules unavailable: {_err}")

    for _name in ['Detect_LSCD', 'Segment_LSCD', 'Pose_LSCD', 'OBB_LSCD', 'Conv_GN', 'Scale']:
        globals()[_name] = _missing_lscd

# YOLO11 head variants import (no custom CUDA build required)
from ultralytics.nn.Head.yolo11_head_variants import (
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    DetectAux,
    Detect_Efficient,
    Detect_LADH,
    Detect_LQE,
    Detect_LSCD_LQE,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_MultiSEAM,
    Detect_RSCD,
    Detect_SEAM,
    OBB_LADH,
    OBB_LQE,
    OBB_LSCD_LQE,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    Pose_LADH,
    Pose_LQE,
    Pose_LSCD_LQE,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Segment_Efficient,
    Segment_LADH,
    Segment_LQE,
    Segment_LSCD_LQE,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
)

_SOEP_IMPORT_ERROR = None
# SOEP 棰堥儴妯″潡瀵煎叆 (灏忕洰鏍囧寮洪噾瀛楀)
try:
    from ultralytics.nn.Neck import (
        SPDConv,
        FGM,
        OmniKernel,
        CSPOmniKernel,
        SNI,
        GSConvE,
        MFM,
        WFU,
        HyperComputeModule,
        MANet,
        MANet_Star,
        MANet_FasterBlock,
        MANet_FasterCGLU,
    )
    SOEP_AVAILABLE = True
except ImportError as _err:
    _SOEP_IMPORT_ERROR = _err
    SOEP_AVAILABLE = False
    LOGGER.warning(f"SOEP neck import failed: {_err}")
    def _missing_soep(*args, **kwargs):
        raise RuntimeError(f"SOEP neck modules unavailable: {_err}")

    for _name in ['SPDConv', 'FGM', 'OmniKernel', 'CSPOmniKernel', 'SNI', 'GSConvE', 'MFM', 'WFU', 'HyperComputeModule', 'MANet', 'MANet_Star', 'MANet_FasterBlock', 'MANet_FasterCGLU']:
        globals()[_name] = _missing_soep

_C3K2_IMPORT_ERROR = None

# C3k2 Extraction 妯″潡瀵煎叆 (C3k2鍙樹綋妯″潡) 鈥?绮剧‘瀛愭ā鍧楀鍏ワ紝閬垮厤鑱氬悎瀵煎叆杩炲甫澶辫触
try:
    from ultralytics.nn.extraction.c3k2_variants import (
        # Batch 1
        C3k2_Faster,
        C3k2_PConv,
        C3k2_ODConv,
        C3k2_Faster_EMA,
        C3k2_DBB,
        C3k2_WDBB,
        C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt,
        C3k2_SCConv,
        C3k2_ScConv,
        C3k2_EMSC,
        C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided,
        C3k2_MSBlock,
        C3k2_EMBC,
        C3k2_EMA,
        # Batch 4
        C3k2_DLKA,
        C3k2_DAttention,
        C3k2_Parc,
        C3k2_DWR,
        C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv,
        C3k2_RFCAConv,
        C3k2_FocusedLinearAttention,
        C3k2_MLCA,
        C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock,
        C3k2_DRB,
        C3k2_DWR_DRB,
        C3k2_AggregatedAtt,
        C3k2_SWC,
        # Batch 7
        C3k2_iRMB,
        C3k2_iRMB_Cascaded,
        C3k2_iRMB_DRB,
        C3k2_iRMB_SWC,
        C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv,
        C3k2_RVB,
        C3k2_RVB_SE,
        C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule,
        C3k2_PPA,
        C3k2_Faster_CGLU,
        C3k2_Star,
        # Batch 10
        C3k2_Star_CAA,
        C3k2_EIEM,
        C3k2_DEConv,
        # Batch 11
        C3k2_gConv,
        C3k2_AdditiveBlock,
        C3k2_AdditiveBlock_CGLU,
        # Batch 12 - 鏂拌縼绉?        C3k2_RetBlock,
        C3k2_Heat,
        C3k2_WTConv,
        C3k2_FMB,
        C3k2_MSMHSA_CGLU,
        C3k2_MogaBlock,
        C3k2_SHSA,
        C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance,
        C3k2_MutilScaleEdgeInformationSelect,
        C3k2_FFCM,
        C3k2_SMAFB,
        C3k2_SMAFB_CGLU,
        C3k2_MSM,
        C3k2_HDRAB,
        C3k2_RAB,
        C3k2_LFE,
        C3k2_IDWC,
        C3k2_IDWB,
        C3k2_CAMixer,
    )
    C3K2_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C3K2_IMPORT_ERROR = _err
    # 涓哄悗缁€昏緫鎻愪緵鍗犱綅锛岄伩鍏?NameError
    C3k2_DAttention = C3k2_Parc = C3k2_FocusedLinearAttention = None
    C3K2_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C3k2 extraction modules import failed: {_err}")

_C2PSA_IMPORT_ERROR = None
try:
    from ultralytics.nn.extraction.c2psa_variants import (
        # Batch 1 - 鍩虹绫诲拰绗竴鎵瑰彉浣?        C2PSA,
        C2fPSA,
        C2BRA,
        BRABlock,
        # Batch 2 - 娉ㄦ剰鍔涙満鍒跺彉浣?        C2CGA,
        CGABlock,
        C2DA,
        DABlock,
        C2DPB,
        DPBlock,
        C2Pola,
        Polalock,
        C2TSSA,
        TSSAlock,
        # Batch 3 - 娉ㄦ剰鍔涙満鍒跺彉浣?+ 褰掍竴鍖栧寮哄彉浣?        C2ASSA,
        ASSAlock,
        C2MSLA,
        MSLAlock,
        C2PSA_DYT,
        PSABlock_DYT,
        C2TSSA_DYT,
        TSSAlock_DYT,
        C2Pola_DYT,
        Polalock_DYT,
        # Batch 4 - FFN澧炲己鍙樹綋
        C2PSA_FMFFN,
        PSABlock_FMFFN,
        C2PSA_CGLU,
        PSABlock_CGLU,
        C2PSA_SEFN,
        PSABlock_SEFN,
        C2PSA_SEFFN,
        PSABlock_SEFFN,
        C2PSA_EDFFN,
        PSABlock_EDFFN,
        # Batch 5 - Mona妯″潡鍖栨敞鎰忓姏褰掍竴鍖?+ 澶嶅悎澧炲己鍙樹綋
        C2PSA_Mona,
        PSABlock_Mona,
        C2TSSA_DYT_Mona,
        TSSAlock_DYT_Mona,
        C2TSSA_DYT_Mona_SEFN,
        TSSAlock_DYT_Mona_SEFN,
        C2TSSA_DYT_Mona_SEFFN,
        TSSAlock_DYT_Mona_SEFFN,
        C2TSSA_DYT_Mona_EDFFN,
        TSSAlock_DYT_Mona_EDFFN,
    )
    C2PSA_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _C2PSA_IMPORT_ERROR = _err
    C2PSA_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"C2PSA extraction modules import failed: {_err}")
    # 鍗犱綅绗︼紝纭繚鍚庣画寮曠敤鏃舵姏鍑烘槑纭敊璇€岄潪 NameError
    def _missing_c2psa(*args, **kwargs):
        raise RuntimeError(f"C2PSA extraction modules unavailable: {_err}")

    for _name in [
        'C2PSA', 'C2fPSA', 'C2BRA', 'BRABlock', 'C2CGA', 'CGABlock', 'C2DA', 'DABlock', 'C2DPB', 'DPBlock',
        'C2Pola', 'Polalock', 'C2TSSA', 'TSSAlock', 'C2ASSA', 'ASSAlock', 'C2MSLA', 'MSLAlock', 'C2PSA_DYT',
        'PSABlock_DYT', 'C2TSSA_DYT', 'TSSAlock_DYT', 'C2Pola_DYT', 'Polalock_DYT', 'C2PSA_FMFFN', 'PSABlock_FMFFN',
        'C2PSA_CGLU', 'PSABlock_CGLU', 'C2PSA_SEFN', 'PSABlock_SEFN', 'C2PSA_SEFFN', 'PSABlock_SEFFN', 'C2PSA_EDFFN',
        'PSABlock_EDFFN', 'C2PSA_Mona', 'PSABlock_Mona', 'C2TSSA_DYT_Mona', 'TSSAlock_DYT_Mona', 'C2TSSA_DYT_Mona_SEFN',
        'TSSAlock_DYT_Mona_SEFN', 'C2TSSA_DYT_Mona_SEFFN', 'TSSAlock_DYT_Mona_SEFFN', 'C2TSSA_DYT_Mona_EDFFN', 'TSSAlock_DYT_Mona_EDFFN',
    ]:
        globals()[_name] = _missing_c2psa

_SPPF_IMPORT_ERROR = None
try:
    from ultralytics.nn.extraction.sppf_base import (
        # Batch 1 - 鏍囧噯 SPPF 鍙樹綋
        SPPF_LSKA,
        # Batch 2 - GOLD-YOLO 鑱氬悎/铻嶅悎
        PyramidPoolAgg,
        PyramidPoolAgg_PCE,
        SimFusion_3in,
        SimFusion_4in,
        AdvPoolFusion,
        # Batch 3 - 娉ㄥ叆/灏忔尝妯″潡
        IFM,
        InjectionMultiSum_Auto_pool,
        WaveletPool,
        WaveletUnPool,
    )
    SPPF_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _SPPF_IMPORT_ERROR = _err
    SPPF_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"SPPF extraction modules import failed: {_err}")
    def _missing_sppf(*args, **kwargs):
        raise RuntimeError(f"SPPF extraction modules unavailable: {_err}")

    for _name in ['SPPF_LSKA', 'PyramidPoolAgg', 'PyramidPoolAgg_PCE', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion', 'IFM', 'InjectionMultiSum_Auto_pool', 'WaveletPool', 'WaveletUnPool']:
        globals()[_name] = _missing_sppf

# C2f Extraction 妯″潡瀵煎叆锛圔atch 01锛夆€?涓嶅仛鑷姩闄嶇骇锛氬鍏ュけ璐ュ簲鐩存帴鏄惧紡鎶ラ敊
from ultralytics.nn.extraction.c2f_variants import (
    C2f_CAMixer,
    C2f_Heat,
    C2f_FMB,
    C2f_MSMHSA_CGLU,
    C2f_MogaBlock,
    C2f_SHSA,
    C2f_SHSA_CGLU,
    C2f_HDRAB,
    C2f_RAB,
    C2f_FFCM,
    C2f_SMAFB,
    C2f_SMAFB_CGLU,
    C2f_AP,
    C2f_CSI,
    C2f_gConv,
    C2f_FCA,
    C2f_FDConv,
    C2f_FDT,
    C2f_FourierConv,
    C2f_GlobalFilter,
    C2f_LSBlock,
    C2f_Strip,
    C2f_StripCGLU,
    C2f_wConv,
    # Batch 03
    C2f_FasterFDConv,
    C2f_FasterSFSConv,
    C2f_Faster_KAN,
    C2f_FAT,
    C2f_SMPCGLU,
    C2f_DBlock,
    # Batch 04 (partial)
    C2f_AdditiveBlock,
    C2f_AdditiveBlock_CGLU,
    C2f_IEL,
    C2f_DTAB,
    C2f_PFDConv,
    C2f_SFSConv,
    C2f_PSFSConv,
    C2f_EBlock,
    # Batch 04 (no-CUDA extras)
    C2f_HFERB,
    C2f_JDPM,
    C2f_ETB,
    C2f_SFHF,
    C2f_MSM,
    C2f_ELGCA,
    C2f_ELGCA_CGLU,
    C2f_LEGM,
    C2f_LFEM,
    C2f_ESC,
    C2f_KAT,
    CSP_MutilScaleEdgeInformationEnhance,
    CSP_MutilScaleEdgeInformationSelect,
    CSP_FreqSpatial,
)

_NECK_IMPORT_ERROR = None
try:
    from ultralytics.nn.Neck import (
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
        PyramidContextExtraction,
        GetIndexOutput,
        RCM,
        FuseBlockMulti,
        DynamicInterpolationFusion,
        ContextGuideFusionModule,
        DASI,
        FocusFeature,
        ParallelAtrousConv,
        CSP_PAC,
        AttentionUpsample,
        AttentionDownsample,
        WFU,
        HyperComputeModule,
        MANet,
        MANet_Star,
        MANet_FasterBlock,
        MANet_FasterCGLU,
    )
    NECK_EXTRACTION_AVAILABLE = True
except ImportError as _err:
    _NECK_IMPORT_ERROR = _err
    NECK_EXTRACTION_AVAILABLE = False
    LOGGER.warning(f"Neck modules import failed: {_err}")
    def _missing_neck(*args, **kwargs):
        raise RuntimeError(f"Neck modules unavailable: {_err}")

    for _name in [
        'GSConv', 'VoVGSCSP', 'AFPN_P345', 'AFPN_P345_Custom', 'AFPN_P2345', 'AFPN_P2345_Custom', 'Zoom_cat', 'ScalSeq', 'DynamicScalSeq', 'HFP', 'SDP', 'SDP_Improved',
        'ChannelAttention_HSFPN', 'ELA_HSFPN', 'CA_HSFPN', 'CAA_HSFPN', 'FSA', 'Add', 'Multiply', 'CrossLayerSpatialAttention',
        'CrossLayerChannelAttention', 'CrossAttentionBlock', 'FreqFusion', 'LocalSimGuidedSampler', 'PSFM', 'Fusion', 'GDSAFusion', 'GLSA', 'CAFMFusion', 'SDI', 'CSPStage', 'BiFusion',
        'OREPANCSPELAN4', 'SBA', 'DPCF', 'HyperACE', 'FullPAD_Tunnel', 'EUCB', 'EUCB_SC', 'MSCB', 'MSCB_SC', 'CSP_MSCB', 'CSP_MSCB_SC',
        'PyramidContextExtraction', 'GetIndexOutput', 'RCM', 'FuseBlockMulti', 'DynamicInterpolationFusion',
        'ContextGuideFusionModule', 'DASI', 'FocusFeature', 'ParallelAtrousConv', 'CSP_PAC',
        'AttentionUpsample', 'AttentionDownsample', 'WFU', 'HyperComputeModule', 'MANet', 'MANet_Star', 'MANet_FasterBlock', 'MANet_FasterCGLU',
    ]:
        globals()[_name] = _missing_neck

# ===== Head Class Sets (align with upstream behavior) =====
DETECT_CLASS: tuple = (
    Detect,
    v8Detect,
    WorldDetect,
    YOLOEDetect,
    v10Detect,
    ImagePoolingAttn,
)
SEGMENT_CLASS: tuple = (
    Segment,
    YOLOESegment,
)
POSE_CLASS: tuple = (Pose,)
OBB_CLASS: tuple = (OBB,)

if LSCD_AVAILABLE:
    DETECT_CLASS = DETECT_CLASS + (Detect_LSCD,)
    SEGMENT_CLASS = SEGMENT_CLASS + (Segment_LSCD,)
    POSE_CLASS = POSE_CLASS + (Pose_LSCD,)
    OBB_CLASS = OBB_CLASS + (OBB_LSCD,)

DETECT_CLASS = DETECT_CLASS + (
    Detect_AFPN_P345,
    Detect_AFPN_P345_Custom,
    Detect_AFPN_P2345,
    Detect_AFPN_P2345_Custom,
    Detect_Efficient,
    DetectAux,
    Detect_SEAM,
    Detect_MultiSEAM,
    Detect_LADH,
    Detect_LSCSBD,
    Detect_LSDECD,
    Detect_RSCD,
    Detect_LQE,
    Detect_LSCD_LQE,
)
SEGMENT_CLASS = SEGMENT_CLASS + (
    Segment_Efficient,
    Segment_LADH,
    Segment_LSCSBD,
    Segment_LSDECD,
    Segment_RSCD,
    Segment_LQE,
    Segment_LSCD_LQE,
)
POSE_CLASS = POSE_CLASS + (
    Pose_LADH,
    Pose_LSCSBD,
    Pose_LSDECD,
    Pose_RSCD,
    Pose_LQE,
    Pose_LSCD_LQE,
)
OBB_CLASS = OBB_CLASS + (
    OBB_LADH,
    OBB_LSCSBD,
    OBB_LSDECD,
    OBB_RSCD,
    OBB_LQE,
    OBB_LSCD_LQE,
)

# ===== Neck Module Class Sets =====
NECK_CLASS: tuple = ()
if NECK_EXTRACTION_AVAILABLE:
    NECK_CLASS = (
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
        SDI,
        Add,
        Multiply,
        CSPStage,
        BiFusion,
        OREPANCSPELAN4,
        SBA,
        DPCF,
        HyperACE,
        FullPAD_Tunnel,
        EUCB,
        EUCB_SC,
        MSCB,
        MSCB_SC,
        CSP_MSCB,
        CSP_MSCB_SC,
    )

# ===== C3k2 Module Class Sets =====
C3K2_CLASS: tuple = (C3k2,)

for _sym in (
    "C3k2_RetBlock",
    "C3k2_Heat",
    "C3k2_WTConv",
    "C3k2_FMB",
    "C3k2_MSMHSA_CGLU",
    "C3k2_MogaBlock",
    "C3k2_SHSA",
    "C3k2_SHSA_CGLU",
    "C3k2_MutilScaleEdgeInformationEnhance",
    "C3k2_MutilScaleEdgeInformationSelect",
    "C3k2_FFCM",
    "C3k2_SMAFB",
    "C3k2_SMAFB_CGLU",
    "C3k2_MSM",
    "C3k2_HDRAB",
    "C3k2_RAB",
    "C3k2_LFE",
):
    if _sym not in globals():
        globals()[_sym] = None

# 鍔ㄦ€佹墿灞?C3k2 Extraction 绯诲垪妯″潡
if C3K2_EXTRACTION_AVAILABLE:
    C3K2_CLASS = C3K2_CLASS + (
        # Batch 1
        C3k2_Faster, C3k2_PConv, C3k2_ODConv, C3k2_Faster_EMA,
        C3k2_DBB, C3k2_WDBB, C3k2_DeepDBB,
        # Batch 2
        C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv,
        C3k2_EMSC, C3k2_EMSCP,
        # Batch 3
        C3k2_ContextGuided, C3k2_MSBlock, C3k2_EMBC, C3k2_EMA,
        # Batch 4
        C3k2_DLKA, C3k2_DAttention, C3k2_Parc, C3k2_DWR, C3k2_RFAConv,
        # Batch 5
        C3k2_RFCBAMConv, C3k2_RFCAConv,
        C3k2_FocusedLinearAttention, C3k2_MLCA, C3k2_AKConv,
        # Batch 6
        C3k2_UniRepLKNetBlock, C3k2_DRB, C3k2_DWR_DRB,
        C3k2_AggregatedAtt, C3k2_SWC,
        # Batch 7
        C3k2_iRMB, C3k2_iRMB_Cascaded, C3k2_iRMB_DRB,
        C3k2_iRMB_SWC, C3k2_DynamicConv,
        # Batch 8
        C3k2_GhostDynamicConv, C3k2_RVB, C3k2_RVB_SE, C3k2_RVB_EMA,
        # Batch 9
        C3k2_PKIModule, C3k2_PPA, C3k2_Faster_CGLU, C3k2_Star,
        # Batch 10
        C3k2_Star_CAA, C3k2_EIEM, C3k2_DEConv,
        # Batch 11
        C3k2_gConv, C3k2_AdditiveBlock, C3k2_AdditiveBlock_CGLU,
        # Batch 12
        C3k2_RetBlock, C3k2_Heat, C3k2_WTConv, C3k2_FMB,
        C3k2_MSMHSA_CGLU, C3k2_MogaBlock, C3k2_SHSA, C3k2_SHSA_CGLU,
        C3k2_MutilScaleEdgeInformationEnhance, C3k2_MutilScaleEdgeInformationSelect, C3k2_FFCM,
        C3k2_SMAFB, C3k2_SMAFB_CGLU,
        C3k2_MSM, C3k2_HDRAB, C3k2_RAB, C3k2_LFE,
    )

# ===== SPPF Variant Class Set =====
SPPF_CLASS: tuple = (SPPF,)
if SPPF_EXTRACTION_AVAILABLE:
    SPPF_CLASS = SPPF_CLASS + (SPPF_LSKA,)

# ===== C2PSA Module Class Sets =====
C2PSA_CLASS: tuple = ()
if C2PSA_EXTRACTION_AVAILABLE:
    C2PSA_CLASS = (C2PSA, C2fPSA) + (
        # Batch 1/2/3 - 娉ㄦ剰鍔涗富骞插彉浣?        C2BRA, C2CGA, C2DA, C2DPB, C2Pola, C2TSSA, C2ASSA, C2MSLA,
        C2PSA_DYT, C2TSSA_DYT, C2Pola_DYT,
        # Batch 4 - FFN 澧炲己
        C2PSA_FMFFN, C2PSA_CGLU, C2PSA_SEFN, C2PSA_SEFFN, C2PSA_EDFFN,
        # Batch 5 - Mona 澶嶅悎澧炲己
        C2PSA_Mona, C2TSSA_DYT_Mona, C2TSSA_DYT_Mona_SEFN, C2TSSA_DYT_Mona_SEFFN, C2TSSA_DYT_Mona_EDFFN,
    )


class BaseModel(torch.nn.Module):
    """
    Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion,
    information display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佽矾鐢卞垵濮嬪寲 =====
        mm_router = None
        mm_routing_enabled = False
        mm_input_sources = None

        # Check if this model has a persistent router (RTDETRDetectionModel)
        if hasattr(self, 'mm_router') and self.mm_router is not None:
            # Use persistent router from model initialization
            mm_router = self.mm_router
            mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
            if profile:
                LOGGER.info("MultiModal: 浣跨敤鎸佷箙鍖杛outer")
        elif MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create temporary router for other model types
                config_dict = getattr(self, 'yaml', None)
                mm_router = MultiModalRouter(config_dict, verbose=profile)
                mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
                if profile:
                    LOGGER.info("MultiModal: 浣跨敤涓存椂router")
            except Exception as e:
                if profile:
                    LOGGER.warning(f"MultiModal routing initialization failed: {e}")
        # ===== MULTIMODAL EXTENSION END =====

        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佸眰绾ц矾鐢卞鐞?=====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====

            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
                if isinstance(m, YOLOEDetect) and hasattr(self, "pe"):
                    m.fuse(self.pe.to(next(self.model.parameters()).device))
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if isinstance(m, heads):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        det_loss_vec, det_items = self.criterion(preds, batch)

        # Optional contrastive branch (enabled only if hooks exist and compute succeeds)
        # Requirements: MULTIMODAL + CONTRAST + registered hooks via 6th field
        mm_hm = getattr(self, 'mm_hook_manager', None)
        has_hooks = False
        if mm_hm is not None:
            # Only enable contrast branch when YAML registered hooks (6th-field) exist
            try:
                has_hooks = bool(mm_hm.has_hooks())
            except Exception:
                has_hooks = False
        use_contrast = (
            self.training
            and MULTIMODAL_AVAILABLE
            and CONTRAST_AVAILABLE
            and has_hooks
        )
        if not use_contrast:
            return det_loss_vec, det_items

        # Read configs from args with safe defaults
        args = getattr(self, 'args', None)
        cfg = ContrastConfig(
            tau=getattr(args, 'contrast_tau', 0.07) if args is not None else 0.07,
            proj_dim=getattr(args, 'contrast_dim', 128) if args is not None else 128,
            lambda_weight=getattr(args, 'contrast_lambda', 0.1) if args is not None else 0.1,
            max_rois_per_image=getattr(args, 'contrast_max_rois', 64) if args is not None else 64,
            share_head=getattr(args, 'contrast_share_head', False) if args is not None else False,
            preferred_stages=tuple(getattr(args, 'contrast_stages', ("P4", "P5", "P3"))) if args is not None else ("P4", "P5", "P3"),
        )

        # Lazy create controller (only when hooks exist)
        if getattr(self, 'mm_contrast_controller', None) is None and use_contrast:
            # Create on correct device to avoid CPU/CUDA mismatch when forward() is first called
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                # Fallback to batch image device if model has no parameters (unlikely)
                img = batch.get('img')
                dev = img.device if isinstance(img, torch.Tensor) else torch.device('cpu')
            self.mm_contrast_controller = ContrastController(cfg).to(dev)

        # Collect hooked features (do not pop to allow external visualization; keep bounded by latest write)
        hook_buffers = mm_hm.collect(pop=False)
        loss_c, stats = self.mm_contrast_controller(hook_buffers, batch)
        # Debug: detect non-finite contrastive loss (when enabled and computed)
        try:
            if loss_c is not None and not torch.isfinite(loss_c):
                from ultralytics.utils import LOGGER as _LOGGER
                _LOGGER.warning(f"[CL][loss] non-finite loss_c detected: {float(loss_c.detach().cpu())}")
        except Exception:
            pass
        if loss_c is None:
            return det_loss_vec, det_items  # no valid pairs this step

        # Compose outputs: scale contrast by lambda for backprop, but report raw value in items
        lambda_w = cfg.lambda_weight
        if det_loss_vec.dim() == 0:
            # safety: ensure vector form
            det_loss_vec = det_loss_vec.unsqueeze(0)
        total_vec = torch.cat([det_loss_vec, (loss_c * lambda_w).unsqueeze(0)], dim=0)
        total_items = torch.cat([det_items, loss_c.detach().unsqueeze(0)], dim=0)
        # Optionally, expose stats via side-effect for loggers (trainer can read from model)
        self._contrast_last_stats = stats
        return total_vec, total_items

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """
    YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass,
    augmented inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 浼犻€掑妯℃€佽矾鐢卞櫒锛堝鏋滃瓨鍦級
        if hasattr(self.model, 'multimodal_router'):
            self.multimodal_router = self.model.multimodal_router
        else:
            self.multimodal_router = None
        # Persist router for runtime ablation/filling so BaseModel forward can reuse it
        self.mm_router = self.multimodal_router if self.multimodal_router is not None else None

        # 瀵规瘮瀛︿範鎺у埗鍣ㄥ欢鍚庡垱寤猴細閬垮厤鍦╥nfo()闃舵瑙﹀彂Lazy鍙傛暟鏈垵濮嬪寲閿欒
        # 瀹為檯鍒涘缓鍦ㄨ缁冨櫒鐨刧et_model闃舵鎴栭娆′娇鐢ㄦ椂锛堣BaseModel.loss锛夈€?
        # 浼犻€扝ookManager锛堝鏋滃瓨鍦級
        self.mm_hook_manager = getattr(self.model, 'mm_hook_manager', None)

        # Build strides
        m = self.model[-1]  # Detect()/Segment()/Pose()/OBB()/...
        heads = DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
        if isinstance(m, heads):  # includes all Detect/Segment/Pose/OBB subclasses (e.g., LSCD variants)
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                seg_pose_obb = SEGMENT_CLASS + POSE_CLASS + OBB_CLASS
                return self.forward(x)[0] if isinstance(m, seg_pose_obb) else self.forward(x)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            _stride_out = _forward(torch.zeros(1, ch, s, s))
            # DetectAux锛氳缁冭緭鍑哄寘鍚緟鍔╁垎鏀紙2*nl 涓壒寰佸浘锛夛紝stride 鍙簲鍩轰簬涓诲垎鏀?nl 涓壒寰佸浘璁＄畻
            if hasattr(m, "dfl_aux") and isinstance(_stride_out, (list, tuple)) and hasattr(m, "nl"):
                nl = int(getattr(m, "nl"))
                if len(_stride_out) >= nl:
                    _stride_out = _stride_out[:nl]
            m.stride = torch.tensor([s / x.shape[-2] for x in _stride_out])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            if hasattr(m, "bias_init") and callable(getattr(m, "bias_init")):
                m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """
        Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        Clip YOLO augmented inference tails.

        Args:
            y (List[torch.Tensor]): List of detection tensors.

        Returns:
            (List[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """
    YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized
    loss computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """
    YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized
    loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """
    YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized
    loss computation for keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """
        Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """
    YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks,
    providing model initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """
        Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """
        Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # ===== MULTIMODAL EXTENSION START - 鎸佷箙鍖朚ultiModalRouter =====
        self.mm_router = None
        self.mm_routing_enabled = False
        self.mm_input_sources = None
        
        if MULTIMODAL_AVAILABLE:
            try:
                from ultralytics.nn.mm import MultiModalRouter
                # Create persistent router with model configuration
                config_dict = getattr(self, 'yaml', None)
                self.mm_router = MultiModalRouter(config_dict, verbose=verbose)
                if verbose:
                    LOGGER.info("RTDETRDetectionModel: persistent MultiModalRouter created")
            except Exception as e:
                if verbose:
                    LOGGER.warning(f"RTDETRDetectionModel: MultiModalRouter鍒濆鍖栧け璐? {e}")
        # ===== MULTIMODAL EXTENSION END =====

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.yaml["nc"], use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佽矾鐢卞垵濮嬪寲 =====
        mm_router = self.mm_router
        mm_routing_enabled, mm_input_sources = mm_router.setup_multimodal_routing(x, profile)
        # ===== MULTIMODAL EXTENSION END =====
        
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佸眰绾ц矾鐢卞鐞?=====
            # Apply multimodal routing if enabled and module has MM attributes
            if mm_routing_enabled and mm_input_sources and mm_router:
                routed_x = mm_router.route_layer_input(x, m, mm_input_sources, profile)
                if routed_x is not None:
                    x = routed_x

            # Check for spatial reset requirement
            if mm_router and hasattr(m, '_mm_spatial_reset') and m._mm_spatial_reset:
                x = mm_router.reset_spatial_input(x, m, mm_input_sources, profile)
            # ===== MULTIMODAL EXTENSION END =====
            
            if profile:
                self._profile_one_layer(m, x, dt)
            # ===== 澶氳緭鍑轰富骞诧紙鍗曟ā鍧楀杈撳嚭锛夋敮鎸侊細涓?RTDETR-main 瀵归綈 =====
            if hasattr(m, "backbone"):
                base_idx = len(y)
                feats = m(x)
                if not isinstance(feats, (list, tuple)):
                    raise TypeError(
                        f"backbone 妯″潡 '{m.__class__.__name__}' 杈撳嚭蹇呴』涓?list/tuple锛屽綋鍓?{type(feats).__name__}"
                    )
                feats = list(feats)
                if len(feats) > 5:
                    raise ValueError(
                        f"backbone 妯″潡 '{m.__class__.__name__}' 杈撳嚭灞傛暟杩囧锛歭en={len(feats)}锛屾湡鏈?=5"
                    )
                for _ in range(5 - len(feats)):
                    feats.insert(0, None)
                for local_i, feat in enumerate(feats):
                    global_i = base_idx + local_i
                    y.append(feat if global_i in self.save else None)
                x = feats[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """
    YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based
    class specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x) or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """
    YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts,
    supporting both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (List[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """
        Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (List[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """
        Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """
        Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (List[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """
        Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """
    YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts,
    providing specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """
    Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through
    model averaging or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """
        Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches the
    error, logs a warning message, and attempts to install the missing module via the check_requirements() function.
    After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR 鉂岋笍 {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR 鉂岋笍 {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """
    Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        weights (str | List[str]): Model weights path(s).
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
    """
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = getattr(model, "task", guess_model_task(model))
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """
    Load a single model weights.

    Args:
        weight (str): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def _validate_and_fill_dea_args(args, c_left):
    """Validate DEA args strictly and auto-fill only the channel when涓篘one.

    鏈熸湜绛惧悕锛欴EA(channel, kernel_size, p_kernel=None, m_kernel=None, reduction=16)
    鍏佽鐨勬渶绠€褰㈠紡锛欴EA([None, kernel_size]) 鎴?DEA([channel, kernel_size])銆?    涓嶅仛鏃у弬鏁伴『搴忕殑鑷姩鏇存锛岄亣鍒颁笉鍚堣鐩存帴鎶ラ敊銆?    """
    a = list(args) if isinstance(args, (list, tuple)) else [args]

    # 鏈€绠€褰㈠紡 [channel/None, kernel_size]
    if len(a) == 2:
        ch, ks = a
        ch = c_left if ch is None else ch
        if not isinstance(ch, int) or not isinstance(ks, int) or ks <= 0:
            raise ValueError(
                f"DEA expects [channel(or None), kernel_size] as minimal form, got {args}"
            )
        return [ch, ks, None, None, 16]

    # 瀹屾暣褰㈠紡锛氳ˉ榻愰暱搴﹀埌5
    while len(a) < 5:
        a.append(None)
    ch, ks, pk, mk, rd = a[:5]
    ch = c_left if ch is None else ch
    if not isinstance(ch, int) or ch <= 0:
        raise ValueError(f"DEA arg[0]=channel must be positive int, got {ch}")
    if not isinstance(ks, int) or ks <= 0:
        raise ValueError(f"DEA arg[1]=kernel_size must be positive int, got {ks}")
    if pk is not None and (not isinstance(pk, (list, tuple)) or len(pk) != 2):
        raise ValueError(f"DEA arg[2]=p_kernel must be 2-list/tuple or None, got {pk}")
    if mk is not None and (not isinstance(mk, (list, tuple)) or len(mk) != 2):
        raise ValueError(f"DEA arg[3]=m_kernel must be 2-list/tuple or None, got {mk}")
    if rd is None:
        rd = 16
    if not isinstance(rd, int) or rd <= 0:
        raise ValueError(f"DEA arg[4]=reduction must be positive int, got {rd}")

    return [ch, ks, pk, mk, rd]


def parse_model(d, ch, verbose=True, dataset_config=None):
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.
        dataset_config (dict, optional): Dataset configuration containing Xch and other multimodal info.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    fusion_mode, node_mode, head_channel = (d.get(x) for x in ("fusion_mode", "node_mode", "head_channel"))

    # 澶氭ā鎬侀厤缃В鏋愶紙浠呭鐞嗗凡杩佺Щ鐨勭粍浠讹級
    multimodal_router = None
    hook_manager = HookManager() if MULTIMODAL_AVAILABLE else None
    hook_parser = MultiModalConfigParser() if MULTIMODAL_AVAILABLE else None
    if MULTIMODAL_AVAILABLE and 'multimodal' in d:
        try:
            # 瑙ｆ瀽澶氭ā鎬侀厤缃?            config_parser = MultiModalConfigParser()
            model_config = config_parser.parse_config(d)

            # 鍒涘缓澶氭ā鎬佽矾鐢卞櫒
            if model_config.get('has_multimodal_layers', False):
                multimodal_router = MultiModalRouter(model_config)
                if verbose:
                    LOGGER.info(f"{colorstr('multimodal:')} Router initialized with {len(model_config.get('input_layers', []))} input layers")
        except Exception as e:
            if verbose:
                LOGGER.warning(f"Failed to initialize multimodal router: {e}")
            multimodal_router = None
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False  # single module multi-output backbone offset flag
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            FourierConv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            # 鏀寔 SPPF 鍙婂叾鍙樹綋
            *SPPF_CLASS,
            # C2PSA 绯诲垪鍦?C2PSA_CLASS 涓粺涓€绠＄悊
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            # C2f 鍙樹綋锛圧TDETRMM mm-mid-c2f-*锛夛細闇€浣滀负 base_modules 鎵嶈兘姝ｇ‘娉ㄥ叆 c1/c2锛堝苟鏀寔 repeats -> n锛?            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            CSP_MutilScaleEdgeInformationEnhance,
            CSP_MutilScaleEdgeInformationSelect,
            CSP_FreqSpatial,
            C2f_BiFocus,
            RepNCSPELAN4,
            RGCSPELAN,
            RepNCSPELAND,  # ELAN followed by dictionary injection (YOLO-RD)
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            ConvNormLayer,  # RTDETR module
        } | set(C3K2_CLASS) | set(C2PSA_CLASS) | set(NECK_CLASS)  # 鍔ㄦ€佹坊鍔犲凡杩佺Щ妯″潡
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C2f_CAMixer,
            C2f_Heat,
            C2f_FMB,
            C2f_MSMHSA_CGLU,
            C2f_MogaBlock,
            C2f_SHSA,
            C2f_SHSA_CGLU,
            C2f_HDRAB,
            C2f_RAB,
            C2f_FFCM,
            C2f_SMAFB,
            C2f_SMAFB_CGLU,
            C2f_AP,
            C2f_CSI,
            C2f_gConv,
            C2f_FCA,
            C2f_FDConv,
            C2f_FDT,
            C2f_FourierConv,
            C2f_GlobalFilter,
            C2f_LSBlock,
            C2f_Strip,
            C2f_StripCGLU,
            C2f_wConv,
            C2f_FasterFDConv,
            C2f_FasterSFSConv,
            C2f_Faster_KAN,
            C2f_FAT,
            C2f_SMPCGLU,
            C2f_DBlock,
            C2f_AdditiveBlock,
            C2f_AdditiveBlock_CGLU,
            C2f_IEL,
            C2f_DTAB,
            C2f_PFDConv,
            C2f_SFSConv,
            C2f_PSFSConv,
            C2f_EBlock,
            C2f_HFERB,
            C2f_JDPM,
            C2f_ETB,
            C2f_SFHF,
            C2f_MSM,
            C2f_ELGCA,
            C2f_ELGCA_CGLU,
            C2f_LEGM,
            C2f_LFEM,
            C2f_ESC,
            C2f_KAT,
            CSP_MutilScaleEdgeInformationEnhance,
            CSP_MutilScaleEdgeInformationSelect,
            CSP_FreqSpatial,
            C2f_BiFocus,
            RGCSPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fCIB,
            A2C2f,
        } | set(C3K2_CLASS) | set(C2PSA_CLASS)  # 鍔ㄦ€佹坊鍔犳墍鏈?C3k2 涓?C2PSA 鍙樹綋
    )
    # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佽矾鐢卞櫒鍒濆鍖?=====
    mm_router = None
    if MULTIMODAL_AVAILABLE:
        try:
            from ultralytics.nn.mm import MultiModalRouter
            # 鏋勫缓閰嶇疆瀛楀吀锛屽寘鍚玠ataset_config
            config_dict = d.copy()
            if dataset_config:
                config_dict['dataset_config'] = dataset_config
            mm_router = MultiModalRouter(config_dict, verbose)
        except Exception as e:
            if verbose:
                LOGGER.warning(f"MultiModal router initialization failed: {e}")
    # ===== MULTIMODAL EXTENSION END =====

    # Generic parser for two-input cross attention/fusion modules (e.g. CTF/MHCA).
    def _parse_two_input_equal_attn(module, f, args, ch, i):
        if isinstance(f, int) or len(f) != 2:
            raise ValueError(f"{module.__name__} expects 2 inputs, got {f} at layer {i}")
        c_left, c_right = ch[f[0]], ch[f[1]]
        # Auto-fill first dim arg if missing/None
        if len(args) == 0:
            args.insert(0, c_left)
        else:
            if args[0] is None:
                args[0] = c_left
        # Module-specific defaults and output channel calc
        if module is MultiHeadCrossAttention:
            # Ensure num_heads exists
            if len(args) < 2:
                args.append(2)
            c2_val = c_left  # tuple outputs with per-branch channels = C
        elif module is CrossTransformerFusion:
            c2_val = c_left * 2  # concat two branches
        else:
            c2_val = c_left
        return c2_val, args

    # 閽堝 C3k2 鍙樹綋鐨勫弬鏁板綊涓€鍖栵紝琛ラ綈缂哄け鐨勫叧閿舰鍙備互鍖归厤涓婃父瀹炵幇
    def _normalize_c3k2_args(module, raw_args):
        # Copy raw args to avoid mutating caller-owned lists.
        args = list(raw_args)
        c2_val = args[0] if args else None

        # 濡傛灉 C3k2 鍙樹綋鏈垚鍔熷鍏ワ紝鍒欐姏鍑烘槑纭敊璇紝閬垮厤闈欓粯闄嶇骇
        if not C3K2_EXTRACTION_AVAILABLE:
            raise RuntimeError(
                "C3k2 extraction modules瀵煎叆澶辫触锛屾棤娉曡В鏋怌3k2灞傚弬鏁般€傚師濮嬪紓甯? "
                f"{_C3K2_IMPORT_ERROR}"
            )

        # fmapsize 鐩稿叧
        if module in (C3k2_DAttention, C3k2_Parc, C3k2_FocusedLinearAttention):
            if len(args) < 2 or isinstance(args[1], bool):
                args = [c2_val, (20, 20)] + args[2:]
            # 纭繚绗笁鍙備负 shortcut
            if len(args) < 3 or not isinstance(args[2], bool):
                args = args[:2] + [True] + args[3:]

        # 鑱氬悎娉ㄦ剰鍔涳細input_resolution & sr_ratio
        elif module is C3k2_AggregatedAtt:
            need_fill = len(args) < 3 or isinstance(args[1], bool) or isinstance(args[2], bool)
            if need_fill:
                input_res = 40 if c2_val is not None and c2_val <= 512 else 20
                sr_ratio = 2 if c2_val is not None and c2_val <= 512 else 1
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, input_res, sr_ratio, shortcut]

        # SWC 浣嶇Щ鍗风Н锛歬ernel_size
        elif module is C3k2_SWC:
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 11 if c2_val is not None and c2_val <= 256 else (9 if c2_val is not None and c2_val <= 512 else 7)
                shortcut = True
                # 鑻ョ浜?绗笁鍙傛暟鏈氨鏄竷灏旓紝瑙嗕綔 shortcut
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        # UniRep 澶ф牳锛歬
        elif module is C3k2_UniRepLKNetBlock:
            if len(args) < 2 or isinstance(args[1], bool):
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [c2_val, 7, shortcut]

        # iRMB 娲剧敓锛氭繁搴﹀嵎绉牳
        elif module in (C3k2_iRMB_DRB, C3k2_iRMB_SWC):
            if len(args) < 2 or isinstance(args[1], bool):
                ks = 13 if c2_val is not None and c2_val <= 256 else (11 if c2_val is not None and c2_val <= 512 else 9)
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                elif len(args) > 2 and isinstance(args[2], bool):
                    shortcut = args[2]
                args = [c2_val, ks, shortcut]

        # PKIModule 澶嶅悎鍙傛暟
        elif module is C3k2_PKIModule:
            need_fill = len(args) < 6 or isinstance(args[1], bool)
            if need_fill or not hasattr(args[1], "__iter__"):
                shortcut = True
                if len(args) > 1 and isinstance(args[1], bool):
                    shortcut = args[1]
                args = [
                    c2_val,
                    (3, 5, 7, 9, 11),  # kernel_sizes
                    1.0,               # expansion
                    True,              # with_caa
                    11,                # caa_kernel_size
                    True,              # add_identity
                    shortcut,
                ]

        return args

    # Modules that return tuple/list outputs but should NOT be treated as backbone multi-output stems.
    tuple_output_modules = frozenset({PyramidContextExtraction, CrossLayerChannelAttention, CrossLayerSpatialAttention})

    for i, layer_config in enumerate(d["backbone"] + d["head"]):  # from, number, module, args, [input_type, hook]
        # ===== MULTIMODAL EXTENSION START - 灞傞厤缃В鏋?=====
        # Parse layer configuration with optional 5th field for multimodal routing
        if mm_router:
            c1, mm_input_source, mm_attributes = mm_router.parse_layer_config(layer_config, i, ch, verbose)
            f, n, m, args = layer_config[:4]  # Extract standard 4 fields
        else:
            # Standard parsing for non-multimodal layers
            if len(layer_config) == 4:
                f, n, m, args = layer_config
            elif len(layer_config) == 5:
                f, n, m, args, _ = layer_config  # Ignore 5th field if no MM router
            else:
                raise ValueError(f"Invalid layer definition at index {i}: expected 4 or 5 elements, got {len(layer_config)}")
            c1 = None
            mm_input_source = None
            mm_attributes = {}
        # ===== MULTIMODAL EXTENSION END =====
        
        # Allow model-level alias for module names, e.g. node_mode: CSP_MSCB.
        if isinstance(m, str) and m in d and isinstance(d[m], str):
            m = d[m]

        try:
            m = (
                getattr(torch.nn, m[3:])
                if "nn." in m
                else getattr(__import__("torchvision").ops, m[16:])
                if "torchvision.ops." in m
                else globals()[m]
            )  # get module
        except KeyError as e:
            raise ImportError(
                f"Module '{m}' is referenced at layer {i} in YAML but is not available in globals(). "
                "Ensure it is properly imported and exported."
            ) from e
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        m_created = False
        m_instance = None
        if m in base_modules:
            # Neck special-cases: these modules do not follow generic [c2, ...] argument pattern.
            if m is Fusion:
                if isinstance(f, int):
                    raise ValueError(f"{m.__name__} expects list inputs, got {f} at layer {i}")
                inc_list = [ch[x] for x in f]
                mode = args[0] if len(args) else (fusion_mode or "bifpn")
                if isinstance(mode, str) and mode in d:
                    mode = d[mode]
                args = [inc_list, mode]
                c2 = sum(inc_list) if mode == "concat" else inc_list[0]
            elif m is GDSAFusion:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                c0, c1 = ch[f[0]], ch[f[1]]
                if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
                args = [c0, c1]
                c2 = c0
            elif m is FreqFusion:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                c_hr, c_lr = ch[f[0]], ch[f[1]]
                if isinstance(c_hr, (list, tuple)) or isinstance(c_lr, (list, tuple)):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c_hr}, {c_lr} at layer {i}")
                c2 = c_hr
                args = [[c_hr, c_lr], *args]
            elif m is CAA_HSFPN:
                c1 = ch[f]
                c2 = c1
                flag = True
                hks = 11
                vks = 11
                if len(args) >= 1:
                    if isinstance(args[0], bool):
                        flag = args[0]
                    else:
                        c2 = args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if len(args) >= 2 and isinstance(args[1], bool):
                    flag = args[1]
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    hks = int(args[2])
                if len(args) >= 4 and isinstance(args[3], (int, float)):
                    vks = int(args[3])
                args = [c1, c2, flag, hks, vks]
            elif m is ChannelAttention_HSFPN:
                c1 = ch[f]
                c2 = c1
                ratio = 4
                flag = True
                if len(args) >= 1:
                    if isinstance(args[0], bool):
                        flag = args[0]
                    elif isinstance(args[0], (int, float)):
                        ratio = int(args[0])
                if len(args) >= 2 and isinstance(args[1], bool):
                    flag = args[1]
                ratio = max(int(ratio), 1)
                args = [c1, ratio, flag]
            elif m is Zoom_cat:
                if isinstance(f, int) or len(f) != 3:
                    raise ValueError(f"{m.__name__} expects 3 inputs, got {f} at layer {i}")
                c2 = sum(ch[x] for x in f)
                args = []
            elif m in {ScalSeq, DynamicScalSeq}:
                if isinstance(f, int) or len(f) != 3:
                    raise ValueError(f"{m.__name__} expects 3 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                c2 = args[0] if len(args) else inc[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [inc, c2]
            elif m is ELA_HSFPN:
                c1 = ch[f]
                c2 = c1
                flag = True
                if len(args) >= 1 and isinstance(args[0], bool):
                    flag = args[0]
                args = [c1, flag]
            elif m is CA_HSFPN:
                c1 = ch[f]
                c2 = c1
                reduction = 8
                flag = True
                if len(args) >= 1:
                    if isinstance(args[0], bool):
                        flag = args[0]
                    elif isinstance(args[0], (int, float)):
                        reduction = int(args[0])
                if len(args) >= 2 and isinstance(args[1], bool):
                    flag = args[1]
                reduction = max(int(reduction), 1)
                args = [c1, reduction, flag]
            elif m is FSA:
                c1 = ch[f]
                c2 = c1
                size = int(args[0]) if len(args) >= 1 and isinstance(args[0], (int, float)) else 512
                ratio = int(args[1]) if len(args) >= 2 and isinstance(args[1], (int, float)) else 10
                args = [c1, size, ratio]
            elif m in {Add, Multiply}:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                c0, c1 = ch[f[0]], ch[f[1]]
                if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
                c2 = c0
                args = []
            elif m is EUCB or m is EUCB_SC:
                c1 = ch[f]
                args = [c1, *args]
                c2 = c1
            elif m is MSCB or m is MSCB_SC:
                c1 = ch[f]
                c2 = args[0] if len(args) else c1
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                kernel_sizes = args[1] if len(args) > 1 else [1, 3, 5]
                stride = args[2] if len(args) > 2 else 1
                expansion_factor = args[3] if len(args) > 3 else 2
                dw_parallel = args[4] if len(args) > 4 else True
                add = args[5] if len(args) > 5 else True
                args = [c1, c2, kernel_sizes, stride, expansion_factor, dw_parallel, add]
            elif m is CSP_MSCB or m is CSP_MSCB_SC:
                c1 = ch[f]
                c2 = args[0] if len(args) else c1
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                kernel_sizes = args[1] if len(args) > 1 else [1, 3, 5]
                shortcut = args[2] if len(args) > 2 else False
                g_arg = args[3] if len(args) > 3 else 1
                e_arg = args[4] if len(args) > 4 else 0.5
                # Consume outer repeat as internal repeat for node_mode-style YAML.
                args = [c1, c2, n, kernel_sizes, shortcut, g_arg, e_arg]
                n = 1
            elif m is SBA:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                if any(isinstance(ci, (list, tuple)) for ci in inc):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
                c2 = args[0] if len(args) else 64
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [inc, c2]
            elif m is DPCF:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                if any(isinstance(ci, (list, tuple)) for ci in inc):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
                c2 = args[0] if len(args) else inc[1]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [inc, c2]
            elif m in {CrossLayerChannelAttention, CrossLayerSpatialAttention}:
                if isinstance(f, int) or len(f) < 2:
                    raise ValueError(f"{m.__name__} expects >=2 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                if any(isinstance(ci, (list, tuple)) for ci in inc):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
                if len(set(inc)) != 1:
                    raise ValueError(f"{m.__name__} expects equal channels, got {inc} at layer {i}")
                in_dim = inc[0]
                if len(args) == 0:
                    args = [in_dim, len(f)]
                else:
                    if args[0] is None:
                        args[0] = in_dim
                    if len(args) == 1:
                        args.append(len(f))
                c2 = inc
            elif m is CrossAttentionBlock:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                if any(isinstance(ci, (list, tuple)) for ci in inc):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
                c2 = args[0] if len(args) else inc[1]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                num_heads = args[1] if len(args) > 1 else 8
                bias = args[2] if len(args) > 2 else False
                args = [inc, c2, num_heads, bias]
            elif m is HyperACE:
                if isinstance(f, int) or len(f) != 3:
                    raise ValueError(f"{m.__name__} expects 3 inputs, got {f} at layer {i}")
                inc = [ch[x] for x in f]
                if any(isinstance(ci, (list, tuple)) for ci in inc):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
                c2 = args[0] if len(args) else inc[1]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                num_hyperedges = args[1] if len(args) > 1 else 8
                dsc3k = args[2] if len(args) > 2 else True
                shortcut = args[3] if len(args) > 3 else False
                e1 = args[4] if len(args) > 4 else 0.5
                e2 = args[5] if len(args) > 5 else 1.0
                context = args[6] if len(args) > 6 else "both"
                channel_adjust = args[7] if len(args) > 7 else False
                args = [inc, c2, n, num_hyperedges, dsc3k, shortcut, e1, e2, context, channel_adjust]
                n = 1
            elif m is FullPAD_Tunnel:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                c0, c1 = ch[f[0]], ch[f[1]]
                if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
                c2 = c0
                args = []
            elif m is PSFM:
                if isinstance(f, int) or len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
                c0, c1 = ch[f[0]], ch[f[1]]
                if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                    raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
                if len(args) == 0:
                    args = [c0]
                elif args[0] is None:
                    args[0] = c0
                c2 = c0
            else:
                # Normalize special C3k2 variant args before c1/c2 injection.
                args = _normalize_c3k2_args(m, args)
                # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬侀€氶亾璁＄畻 =====
                # Use multimodal router computed c1 if available, otherwise use standard logic
                if mm_input_source and c1 is not None:
                    # For multimodal layers, c1 is computed by router, c2 from args
                    c2 = args[0]
                    # c1 is already set by mm_router.parse_layer_config()
                else:
                    c1, c2 = ch[f], args[0]
                # ===== MULTIMODAL EXTENSION END =====
                if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                if m is C2fAttn:  # set 1) embed channels and 2) num heads
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)  # number of repeats
                    n = 1
                if m in C3K2_CLASS:  # for M/L/X sizes - 鏀寔鎵€鏈塁3k2鍙樹綋
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if m is A2C2f:
                    legacy = False
                    if scale in "lx":  # for L/X sizes
                        args.extend((True, 1.2))
                if m is C2fCIB:
                    legacy = False
        elif m in {
            AIFI,
            AIFI_LPE,
            AIFI_RepBN,
            AIFI_SEFN,
            AIFI_Mona,
            AIFI_DyT,
            AIFI_SEFFN,
            AIFI_EDFFN,
            TransformerEncoderLayer_LocalWindowAttention,
            TransformerEncoderLayer_DAttention,
            TransformerEncoderLayer_HiLo,
            TransformerEncoderLayer_EfficientAdditiveAttnetion,
            TransformerEncoderLayer_AdditiveTokenMixer,
            TransformerEncoderLayer_MSMHSA,
            TransformerEncoderLayer_DHSA,
            TransformerEncoderLayer_DPB,
            TransformerEncoderLayer_Pola,
            TransformerEncoderLayer_TSSA,
            TransformerEncoderLayer_ASSA,
            TransformerEncoderLayer_MSLA,
            TransformerEncoderLayer_Pola_SEFN,
            TransformerEncoderLayer_ASSA_SEFN,
            TransformerEncoderLayer_ASSA_SEFN_Mona,
            TransformerEncoderLayer_Pola_SEFN_Mona,
            TransformerEncoderLayer_ASSA_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFN_Mona_DyT,
            TransformerEncoderLayer_Pola_SEFFN_Mona_DyT,
            TransformerEncoderLayer_Pola_EDFFN_Mona_DyT,
        }:
            args = [ch[f], *args]
        elif m in {
            TimmBackbone,
            convnextv2_atto,
            convnextv2_femto,
            convnextv2_pico,
            convnextv2_nano,
            convnextv2_tiny,
            convnextv2_base,
            convnextv2_large,
            convnextv2_huge,
            repvit_m0_9,
            repvit_m1_0,
            repvit_m1_1,
            repvit_m1_5,
            repvit_m2_3,
            efficientformerv2_s0,
            efficientformerv2_s1,
            efficientformerv2_s2,
            efficientformerv2_l,
            EfficientViT_M0,
            EfficientViT_M1,
            EfficientViT_M2,
            EfficientViT_M3,
            EfficientViT_M4,
            EfficientViT_M5,
            SwinTransformer_Tiny,
        }:
            # 绾︽潫锛氬崟妯″潡澶氳緭鍑轰富骞插繀椤诲嚭鐜板湪 backbone 棣栧眰锛屽惁鍒欑储寮曚綋绯讳細浜х敓姝т箟
            if i != 0:
                raise RuntimeError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} must be the first backbone layer (index 0) "
                    "when used as a single-module multi-output backbone."
                )
            if n != 1:
                raise ValueError(
                    f"{m.__name__ if hasattr(m, '__name__') else m} does not support repeats > 1."
                )
            # 娉ㄥ叆杈撳叆閫氶亾锛堜紭鍏堜娇鐢?router 瑙ｆ瀽鍑虹殑 c1锛屼互鏀寔 Dual=6ch 绛夊妯℃€佽緭鍏ワ級
            in_ch = c1 if c1 is not None else ch[f]
            args = [in_ch, *args]
            m_instance = m(*args)
            m_created = True
            c2 = list(getattr(m_instance, "channel"))
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # ===== GOLD-YOLO / 鎻愬彇绫绘睜鍖?铻嶅悎妯″潡 =====
        elif SPPF_EXTRACTION_AVAILABLE and m in frozenset({SimFusion_4in, AdvPoolFusion}):
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is SimFusion_3in:
            # SimFusion_3in([c0,c1,c2], out)
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif SPPF_EXTRACTION_AVAILABLE and m is IFM:
            # IFM(c1, [o1,o2,...], embed_dim_p=96, fuse_block_num=3)
            c1 = ch[f]
            c2 = sum(args[0])
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is InjectionMultiSum_Auto_pool:
            # InjectionMultiSum_Auto_pool(inp, oup, global_inp, flag), f=[local, global]
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg:
            # PyramidPoolAgg(inc=sum(inputs), ouc=args[0], stride, pool_mode='torch')
            c2 = args[0]
            args = [sum(ch[x] for x in f), *args]
        elif SPPF_EXTRACTION_AVAILABLE and m is PyramidPoolAgg_PCE:
            c2 = sum(ch[x] for x in f)
        elif SPPF_EXTRACTION_AVAILABLE and m is WaveletPool:
            # 灏忔尝涓嬮噰鏍凤紝閫氶亾*4
            c2 = ch[f] * 4
        elif m is DEA:
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            args = _validate_and_fill_dea_args(args, c_left)
            c2 = c_left
        elif m is MCFGatedFusion:
            if isinstance(f, int) or len(f) < 2:
                raise ValueError(f"{m.__name__} expects >=2 inputs, got {f} at layer {i}")
            mode = args[0] if len(args) > 0 else "add"
            k = args[1] if len(args) > 1 else 1
            c_out = args[2] if len(args) > 2 and args[2] else None
            main_idx = args[3] if len(args) > 3 else 0
            aux_idx = args[4] if len(args) > 4 else 1
            zero_init = args[5] if len(args) > 5 else True
            use_bn = args[6] if len(args) > 6 else False
            act = args[7] if len(args) > 7 else True

            c_main = ch[f[main_idx]]
            c_aux = ch[f[aux_idx]]
            c_out = c_out or c_main
            if c_out != nc:
                c_out = make_divisible(min(c_out, max_channels) * width, 8)
            args = [c_main, c_aux, c_out, mode, k, 1, None, 1, main_idx, aux_idx, zero_init, use_bn, act]
            c2 = c_main
        elif m in {MultiScaleGatedAttn, FusionBiFPN}:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            if len(args) == 0:
                args = [[c_left, c_right]]
            elif args[0] is None:
                args[0] = [c_left, c_right]
            c2 = min(c_left, c_right)
        elif m in frozenset({FeatureFusion, FCM, FCMFeatureFusion, ConvMixFusion, CAM, SEFN, FusionConvMSAA, MSC, SpatialDependencyPerception, CGAFusion, CMXFusion, SuperYOLOFusion, SigmaFusionBlock, TarDALFusion}):
            # Expect exactly two inputs; output channels follow the left branch
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            # Auto infer dim if not provided (None or missing)
            if len(args) == 0:
                args.insert(0, c_left)
            elif args[0] is None:
                args[0] = c_left
            c2 = c_left
        elif m is MetaFeatureFusion:
            # MetaFeatureFusion(c1, c2): c1=fusion branch channels, c2=semantic branch channels
            # output channels = c1 (residual fusion, Fout = Fuj + Ftj)
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            args = [c_left, c_right]
            c2 = c_left
        elif m is CENFusion:
            # CENFusion(c, p=0.5): c=input channels (auto-injected), p=swap ratio from YAML args[0]
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left = ch[f[0]]
            p = args[0] if len(args) > 0 else 0.5
            args = [c_left, p]
            c2 = c_left
        elif m is MambaDFuseBlock:
            # MambaDFuseBlock(c1, c2, p=0.5): c1=c2=input channels, p=exchange ratio from YAML args[0]
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            p = args[0] if len(args) > 0 else 0.5
            args = [c_left, c_right, p]
            c2 = c_left
        elif m is PIAFusionBlock:
            # PIAFusionBlock(c1, c2): illumination-aware + CMDAF, output = c1 + c2 (cat)
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            args = [c_left, c_right]
            c2 = c_left + c_right  # 输出为拼接通道数
        elif m is CDDFusion:
            # CDDFusion(ch=[c1,c2], c_out=256): dual-branch feature decomposition fusion
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            c_out_val = args[0] if len(args) > 0 and args[0] is not None else 256
            args = [[c_left, c_right], c_out_val]
            c2 = c_out_val
        elif m is ScalarGate:
            # Dual-path mode: [[rgb, ir], 1, ScalarGate, []] -> global scalar modal fusion gate
            # Single-path mode: [-1, 1, ScalarGate, []]      -> global scalar SE attention on fused feature
            if isinstance(f, int):
                # single-path: channel-preserving global scalar attention
                c2 = ch[f]
            else:
                # dual-path: modal fusion, output = left branch channels
                if len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 1 or 2 inputs, got {f} at layer {i}")
                c2 = ch[f[0]]
            if len(args) == 0:
                args.insert(0, c2)
            elif args[0] is None:
                args[0] = c2
        elif m is IIA:
            # Single-input directional attention; channel-preserving, inject c1 so GFLOPs profile works
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]
            if len(args) == 0:
                args.insert(0, c2)
            elif args[0] is None:
                args[0] = c2
        elif m is ChannelGate:
            # Dual-path mode: [[rgb, ir], 1, ChannelGate, []] -> modal fusion gate
            # Single-path mode: [-1, 1, ChannelGate, []]      -> SE channel attention on fused feature
            if isinstance(f, int):
                # single-path: channel-preserving SE attention
                c2 = ch[f]
            else:
                # dual-path: modal fusion, output = left branch channels
                if len(f) != 2:
                    raise ValueError(f"{m.__name__} expects 1 or 2 inputs, got {f} at layer {i}")
                c2 = ch[f[0]]
            if len(args) == 0:
                args.insert(0, c2)
            elif args[0] is None:
                args[0] = c2
        elif m is SEChannelAttention:
            # Single-path SE channel attention: channel-preserving, auto-infer channels from upstream
            if not isinstance(f, int):
                raise ValueError(f"{m.__name__} expects a single input, got {f} at layer {i}")
            c2 = ch[f]
            if len(args) == 0:
                args.insert(0, c2)
            elif args[0] is None:
                args[0] = c2
            # optional: args[1] = reduction ratio (default 16, kept as-is if provided)
        elif m is PST:
            # Pyramid Sparse Transformer expects (x, upper_feat) with upper_feat at 1/2 resolution.
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs (x, upper_feat), got {f} at layer {i}")
            c_left, c_up = ch[f[0]], ch[f[1]]
            if len(args) < 1:
                raise ValueError(f"{m.__name__} requires args [c2, mlp_ratio?, e?, k?], got {args} at layer {i}")
            c_out = args[0]
            mlp_ratio = args[1] if len(args) > 1 else 2.0
            e = args[2] if len(args) > 2 else 0.5
            k = args[3] if len(args) > 3 else 0
            # signature: PST(c1, c_up, c2, n=1, mlp_ratio=..., e=..., k=...)
            args = [c_left, c_up, c_out, n, mlp_ratio, e, k]
            c2 = c_out
            n = 1
        elif m is SpatialPriorModuleLite:
            if isinstance(f, (list, tuple)) and len(f) != 1:
                raise ValueError(f"{m.__name__} expects 1 input, got {f} at layer {i}")
            if len(args) < 3:
                raise ValueError(
                    f"{m.__name__} requires args [inplanes, (C8,C16,C32), in_chans][, use_bn], "
                    f"got layer {i} args: {args}"
                )
            embed = args[1]
            if not isinstance(embed, (list, tuple)) or len(embed) != 3:
                raise ValueError(
                    f"{m.__name__} second arg must be a 3-tuple/list (C8,C16,C32), got: {embed}"
                )
            try:
                c2 = int(embed[0])
            except Exception as e:
                raise ValueError(f"{m.__name__} embed_dims[0] must be int, got: {embed[0]}") from e
        elif m in frozenset({CrossTransformerFusion, MultiHeadCrossAttention}):
            c2, args = _parse_two_input_equal_attn(m, f, args, ch, i)
        elif m is ConvFFN_GLU:
            # Standalone conv-ffn with GLU gate. By default assumes input is 2C and output is C.
            # If args not provided, set [in_channels=c_in, out_channels=c_in // 2]
            c_in = ch[f]
            if len(args) < 2:
                # [in_channels, out_channels]
                args = [c_in, max(c_in // 2, 1), *args]
            else:
                # fill None placeholders
                if args[0] is None:
                    args[0] = c_in
                if args[1] is None:
                    args[1] = max(c_in // 2, 1)
            c2 = args[1]
        elif m in (DETECT_CLASS + SEGMENT_CLASS + POSE_CLASS + OBB_CLASS):
            # 涓烘娴?鍒嗗壊/濮挎€?鏃嬭浆澶存敞鍏ュ悇灞傝緭鍏ラ€氶亾鍒楄〃
            args.append([ch[x] for x in f])
            # 鍒嗗壊澶寸殑閫氱敤閫氶亾缂╂斁锛堜笌涓婃父淇濇寔涓€鑷达級
            if m in SEGMENT_CLASS:
                # 鏌愪簺鍒嗗壊瀹炵幇浼氫娇鐢ㄧ涓変釜浣嶇疆浣滀负閫氶亾鐩稿叧瓒呭弬锛堝鍘熺敓 Segment 鐨?npr/c4锛夛紝淇濇寔涓庝笂娓镐竴鑷寸殑缂╂斁
                if len(args) > 2 and isinstance(args[2], (int, float)):
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            for cls_ in (Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB):
                if m is cls_:
                    m.legacy = legacy
                    break

            # ===== LSCD 绯诲垪涓撻」缂╂斁锛氭寜涓婃父鍋氭硶瀵?hidc 绛夐€氶亾鍨嬪弬鏁拌繘琛屽搴︾缉鏀?=====
            if LSCD_AVAILABLE:
                if m is Detect_LSCD and len(args) > 1 and isinstance(args[1], (int, float)):
                    # Detect_LSCD(nc, hidc, ch)
                    args[1] = make_divisible(min(args[1], max_channels) * width, 8)
                elif m is Segment_LSCD and len(args) > 3 and isinstance(args[3], (int, float)):
                    # Segment_LSCD(nc, nm, npr, hidc, ch) -> 缂╂斁 hidc
                    args[3] = make_divisible(min(args[3], max_channels) * width, 8)
                elif m is Pose_LSCD and len(args) > 2 and isinstance(args[2], (int, float)):
                    # Pose_LSCD(nc, kpt_shape, hidc, ch) -> 缂╂斁 hidc
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                elif m is OBB_LSCD and len(args) > 2 and isinstance(args[2], (int, float)):
                    # OBB_LSCD(nc, ne, hidc, ch) -> 缂╂斁 hidc
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)

            # ===== YOLO11 妫€娴嬪ご鍙樹綋涓撻」缂╂斁/瑙ｆ瀽锛堟棤闇€缂栬瘧锛?====
            # 1) AFPN 绯诲垪锛欴etect_AFPN_*(nc, hidc, [block_type], ch)
            if m in (Detect_AFPN_P345, Detect_AFPN_P2345) and len(args) > 1 and isinstance(args[1], (int, float)):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            elif m in (Detect_AFPN_P345_Custom, Detect_AFPN_P2345_Custom):
                # args: [nc, hidc, block_type(str|cls), ch]
                if len(args) > 1 and isinstance(args[1], (int, float)):
                    args[1] = make_divisible(min(args[1], max_channels) * width, 8)
                if len(args) > 2 and isinstance(args[2], str):
                    if args[2] not in globals():
                        raise ValueError(
                            f"{m.__name__} block_type='{args[2]}' is not registered in tasks.py globals()."
                        )
                    args[2] = globals()[args[2]]

            # 2) 鍏变韩鍗风Н澶村鏃忥細Detect_LSCSBD/LSDECD/RSCD/LSCD_LQE -> 缂╂斁 hidc
            if m in (Detect_LSCSBD, Detect_LSDECD, Detect_RSCD, Detect_LSCD_LQE) and len(args) > 1 and isinstance(args[1], (int, float)):
                args[1] = make_divisible(min(args[1], max_channels) * width, 8)
            elif m in (Segment_LSCSBD, Segment_LSDECD, Segment_RSCD, Segment_LSCD_LQE) and len(args) > 3 and isinstance(args[3], (int, float)):
                # Segment_*(nc, nm, npr, hidc, ch)
                args[3] = make_divisible(min(args[3], max_channels) * width, 8)
            elif m in (Pose_LSCSBD, Pose_LSDECD, Pose_RSCD, Pose_LSCD_LQE) and len(args) > 2 and isinstance(args[2], (int, float)):
                # Pose_*(nc, kpt_shape, hidc, ch)
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            elif m in (OBB_LSCSBD, OBB_LSDECD, OBB_RSCD, OBB_LSCD_LQE) and len(args) > 2 and isinstance(args[2], (int, float)):
                # OBB_*(nc, ne, hidc, ch)
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is DConv:
            # RD: channel-preserving injection block, supports (c1, alpha=0.8, atoms=512)
            # Flexible YAML args: [] or [atoms] or [alpha] or [atoms, alpha]
            c1 = ch[f]
            c2 = ch[f]
            alpha = 0.8
            atoms = 512
            if len(args) == 1:
                if isinstance(args[0], (int, torch.Tensor)):
                    atoms = int(args[0])
                else:
                    alpha = float(args[0])
            elif len(args) >= 2:
                # assume [atoms, alpha]
                atoms = int(args[0])
                alpha = float(args[1])
            args = [c1, alpha, atoms, *args[2:]]
        elif m is TorchVision:
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        elif m is Index:
            # Select an element from a tuple/list output (e.g., from FCM)
            # Output channels remain the same as input referenced by 'f'
            c2 = ch[f]
            # keep args as-is: [index]
        elif m is Blocks:
            # 鐗规畩澶勭悊Blocks妯″潡锛歔ch_out, block_type, block_nums, stage_num, act, variant]
            block_type = globals()[args[1]]  # 灏嗗瓧绗︿覆杞崲涓虹被
            c1, c2 = ch[f], args[0] * block_type.expansion
            args = [c1, args[0], block_type, *args[2:]]
        
        # === SOEP Neck妯″潡澶勭悊 ===
        elif m is CSPOmniKernel:
            # CSPOmniKernel: 鍙渶瑕乨im鍙傛暟锛屼粠杈撳叆閫氶亾鑷姩鎺ㄦ柇
            c2 = ch[f]
            args = [c2]
        elif m is SPDConv:
            # SPDConv: 绌洪棿鍒版繁搴﹀嵎绉紝闇€瑕?inc 鍜?ouc锛屽彲閫?dimension锛堥粯璁?锛?            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            dim_arg = args[1] if len(args) > 1 else 1
            args = [c1, c2, dim_arg]
        elif m is MFM:
            # MFM: 澶氬昂搴︾壒寰佽皟鍒讹紝闇€瑕乮nc鍒楄〃鍜宒im鍙傛暟
            # args搴旇鏄痆dim, reduction]锛宨nc浠巉鑷姩鎺ㄦ柇
            if len(args) < 1:
                raise ValueError(f"MFM requires at least 1 arg (dim), got {args}")
            inc = [ch[x] for x in f]
            dim = args[0]
            c2 = dim
            # args淇濇寔涓篬inc, dim, reduction(鍙€?]
            if len(args) >= 2:
                args = [inc, dim, args[1]]
            else:
                args = [inc, dim]
        elif m in {MANet, MANet_Star, MANet_FasterBlock, MANet_FasterCGLU}:
            # MANet variants: [c2, shortcut, p, kernel_size]
            c1 = ch[f]
            c2 = args[0] if len(args) > 0 else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, n, *args[1:]]
            n = 1
        elif m is WFU:
            # WFU: [x_big, x_small] -> out with x_big channels/resolution
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            inc = [ch[x] for x in f]
            if any(isinstance(ci, (list, tuple)) for ci in inc):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
            c2 = inc[0]
            args = [inc]
        # === C3k2 鍙樹綋锛堥渶瑕佸弬鏁伴噸鎺掞級 ===
        elif C3K2_EXTRACTION_AVAILABLE and m in (C3k2_DAttention,):
            # 鍏煎 YAML 鍐欐硶 [c2, c3k?(bool), e?(float)]
            # 鐩爣绛惧悕: (c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True)
            # 娉ㄦ剰锛歱arse_model 宸茬敤澶栧眰 n 閲嶅妯″潡锛屽唴閮?n 淇濇寔榛樿1
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            c3k_flag = False
            e_val = 0.5
            if len(args) > 1 and isinstance(args[1], bool):
                c3k_flag = args[1]
            if len(args) > 2 and isinstance(args[2], (int, float)):
                e_val = float(args[2])
            # 涓嶆樉寮忎紶鍏?fmapsize锛堜繚鎸佷负 None锛屽唴閮ㄨ嚜閫傚簲锛夛紝涓嶄紶 g/shortcut锛堜娇鐢ㄩ粯璁わ級
            args = [c1, c2, 1, None, c3k_flag, e_val]
        elif m is SNI:
            # SNI: 杞渶杩戦偦鎻掑€硷紝涓婇噰鏍峰眰锛岄€氶亾鏁颁繚鎸佷笉鍙橈紝浠呮帴鏀?up_f 鍙傛暟
            # 淇濇寔 YAML 涓殑 args锛堝 [2]锛変笉鍙橈紝涓嶆敞鍏ラ€氶亾鍙傛暟
            c1, c2 = ch[f], ch[f]
        elif m is GSConvE:
            c1, c2 = ch[f], args[0] if len(args) > 0 else ch[f]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        # === CGRFPN blocks ===
        elif m is PyramidContextExtraction:
            if isinstance(f, int):
                raise ValueError(f"{m.__name__} expects list inputs, got {f} at layer {i}")
            c_in = [ch[x] for x in f]
            c2 = c_in
            args = [c_in, *args]
        elif m is GetIndexOutput:
            if not isinstance(ch[f], (list, tuple)):
                raise ValueError(f"{m.__name__} expects list-like input channels at layer {i}, got {ch[f]}")
            c2 = ch[f][args[0]]
        elif m is RCM:
            c2 = ch[f]
            args = [c2, *args]
        elif m is DynamicInterpolationFusion:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c2 = ch[f[0]]
            args = [[ch[x] for x in f], *args]
        elif m is FuseBlockMulti:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c2 = ch[f[0]]
            args = [c2, *args]
        elif m is ContextGuideFusionModule:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c0, c1 = ch[f[0]], ch[f[1]]
            if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
            c2 = c1 * 2
            args = [[c0, c1], *args]
        elif m is CAFMFusion:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c0, c1 = ch[f[0]], ch[f[1]]
            if isinstance(c0, (list, tuple)) or isinstance(c1, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c0}, {c1} at layer {i}")
            if c0 != c1:
                raise ValueError(f"{m.__name__} expects equal channels, got {c0} and {c1} at layer {i}")
            heads = args[0] if len(args) > 0 else 8
            args = [c0, heads]
            c2 = c0
        elif m is FreqFusion:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_hr, c_lr = ch[f[0]], ch[f[1]]
            if isinstance(c_hr, (list, tuple)) or isinstance(c_lr, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c_hr}, {c_lr} at layer {i}")
            c2 = c_hr
            args = [[c_hr, c_lr], *args]
        elif m is CFC_CRB:
            c1 = ch[f]
            if isinstance(c1, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor input, got channels {c1} at layer {i}")
            c2 = max(c1 // 2, 1)
            args = [c1, *args]
        elif m is SFC_G2:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            if isinstance(c_left, (list, tuple)) or isinstance(c_right, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c_left}, {c_right} at layer {i}")
            c2 = c_left
            args = [[c_left, c_right]]
        elif m is PSFM:
            if isinstance(f, int) or len(f) != 2:
                raise ValueError(f"{m.__name__} expects 2 inputs, got {f} at layer {i}")
            c_left, c_right = ch[f[0]], ch[f[1]]
            if isinstance(c_left, (list, tuple)) or isinstance(c_right, (list, tuple)):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {c_left}, {c_right} at layer {i}")
            if len(args) == 0:
                args = [c_left]
            elif args[0] is None:
                args[0] = c_left
            c2 = c_left
        elif m is DASI:
            if isinstance(f, int) or len(f) < 2:
                raise ValueError(f"{m.__name__} expects >=2 inputs, got {f} at layer {i}")
            inc = [ch[x] for x in f]
            if any(isinstance(ci, (list, tuple)) for ci in inc):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
            c2 = args[0] if len(args) > 0 else inc[len(inc) // 2]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [inc, c2]
        elif m is FocusFeature:
            if isinstance(f, int) or len(f) < 3:
                raise ValueError(f"{m.__name__} expects >=3 inputs, got {f} at layer {i}")
            inc = [ch[x] for x in f]
            if any(isinstance(ci, (list, tuple)) for ci in inc):
                raise ValueError(f"{m.__name__} expects tensor inputs, got channels {inc} at layer {i}")
            # args: [kernel_sizes?, e?]
            e = 0.5
            if len(args) == 1 and isinstance(args[0], (float, int)):
                e = float(args[0])
            elif len(args) >= 2 and isinstance(args[1], (float, int)):
                e = float(args[1])
            c2 = max(int(inc[1] * e), 1) * 3
            args = [inc, *args]
        elif m is ParallelAtrousConv:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is CSP_PAC:
            c1 = ch[f]
            c2 = args[0] if len(args) > 0 else c1
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {AttentionUpsample, AttentionDownsample}:
            c1 = ch[f]
            c2 = c1
            args = [c1]

        # NNexpend [disabled by default]
        # Extra modules 鎵╁睍妯″潡澶勭悊锛堝凡榛樿绂佺敤锛屼繚鐣欏崰浣嶆敞閲婏級

        # elif m is PST:
        #     c1,c_up,c2 = ch[f[0]],ch[f[1]],args[0]
        #     c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     args = [c1,c_up,c2, *args[1:]]
        #     args.insert(3,n)
        #     n = 1
        # elif m is C3k2_KW:  # C3k2鏍镐粨搴撴ā鍧?- 鏉ヨ嚜extra_modules
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     # 鍒濆鍖栨牳浠撳簱绠＄悊鍣?        #     if 'warehouse_manager' not in locals():
        #         from ultralytics.nn.extra_modules.kernel_warehouse import get_warehouse_manager
        #         warehouse_manager = get_warehouse_manager()
        #     args = [c1, c2, *args[1:]]
        #     # 鍦ㄧ壒瀹氫綅缃彃鍏ュ眰鍚嶇О鍜屼粨搴撶鐞嗗櫒
        #     args.insert(2, f'layer{i}')
        #     args.insert(2, warehouse_manager)
        #     if n > 1:
        #         args.insert(4, n)  # 鐢变簬鎻掑叆浜嗗弬鏁帮紝璋冩暣浣嶇疆
        #         n = 1
        # elif m in {C3k2_DySnakeConv, C3k2_OREPA, C3k2_REPVGGOREPA,
        #           C3k2_RFAConv, C3k2_RFCBAMConv, C3k2_RFCAConv,
        #           C3k2_VSS, C3k2_wConv}:  # 鍏朵粬C3k2鍙樹綋 - 鏉ヨ嚜extra_modules
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     # 鏍囧噯C3k2鍙傛暟澶勭悊
        #     args = [c1, c2, *args[1:]]
        #     # 鐗瑰畾鍙樹綋鐨勭壒娈婂鐞?        #     if m is C3k2_DySnakeConv:
        #         # DySnakeConv鍙兘闇€瑕佺壒娈婄殑閫氶亾澶勭悊
        #         # 浣嗗浜嶤3k2鍙樹綋锛屼繚鎸佹爣鍑嗗鐞?        #         pass
        #     # 涓篊3k2鍧楁坊鍔爊鍙傛暟
        #     if n > 1:
        #         args.insert(2, n)
        #         n = 1
        #     # M/L/X灏哄鐨勭壒娈婂鐞?        #     if scale in "mlx":
        #         # C3k2_wConv鏈夌壒娈婄殑鍙傛暟澶勭悊
        #         if m is C3k2_wConv:
        #             if len(args) > 0 and isinstance(args[-1], bool):
        #                 args[-1] = True
        #             elif len(args) > 1:
        #                 args[-2] = True
        #         else:
        #             # 鍏朵粬C3k2鍙樹綋浣跨敤鏍囧噯鐨刟rgs[3]澶勭悊
        #             if len(args) > 3:
        #                 args[3] = True

        # 鏍囧噯鍗风Н妯″潡澶勭悊 - 鏉ヨ嚜extra_modules锛堥粯璁ょ鐢級
        # elif m in {RFAConv, RFCBAMConv, RFCAConv,  # RFA绯诲垪鍗风Н
        #           VSSBlock_YOLO, XSSBlock,  # Mamba VSS/XSS鍧?        #           CSP_FreqSpatial,  # 棰戠┖闂碈SP妯″潡
        #           FeaturePyramidSharedConv,  # 鐗瑰緛閲戝瓧濉斿叡浜嵎绉?        #           DSConv_YOLO13, wConv2d}:  # YOLO13绯诲垪鍗风Н
        #     c1, c2 = ch[f], args[0]
        #     if c2 != nc:
        #         c2 = make_divisible(min(c2, max_channels) * width, 8)
        #     args = [c1, c2, *args[1:]]
        #     # 鍙湁XSSBlock鍜孋SP_FreqSpatial鏀寔閲嶅鍙傛暟
        #     if m in {XSSBlock, CSP_FreqSpatial} and n > 1:
        #         args.insert(2, n)
        #         n = 1

        # 娉ㄦ剰鍔涙満鍒舵ā鍧楀鐞?- 鏉ヨ嚜extra_modules锛堥粯璁ょ鐢級
        # elif m in {EMA, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw,
        #           TripletAttention, CoordAtt,
        #         #   CBAM,
        #           BAMBlock, LSKBlock, ScConv, LAWDS, EMSConv, EMSConvP,
        #           SEAttention, CPCA, Partial_conv3, FocalModulation, EfficientAttention, MPCA, deformable_LKA,
        #           EffectiveSEModule, LSKA, SegNext_Attention, DAttention, MLCA, TransNeXt_AggregatedAttention,
        #           FocusedLinearAttention, LocalWindowAttention, ChannelAttention_HSFPN, ELA_HSFPN, CA_HSFPN, CAA_HSFPN,
        #           DySample, CARAFE, CAA, ELA, CAFM, AFGCAttention, EUCB, EfficientChannelAttention,
        #           ContrastDrivenFeatureAggregation, FSA, AttentiveLayer, EUCB_SC}:  # 鎵€鏈夋敞鎰忓姏鏈哄埗妯″潡
        #     c2 = ch[f]
        #     args = [c2, *args]

        # NNexpend锛堥粯璁ょ鐢級

        # elif m in {HAFB}:
        #     if args[0] == 'head_channel':
        #         args[0] = d[args[0]]
        #     c1 = [ch[x] for x in f]
        #     c2 = make_divisible(min(args[0], max_channels) * width, 8)
        #     args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f] if isinstance(f, int) else args[0]

        # ===== 鍗曟ā鍧楀杈撳嚭涓诲共锛歝2 涓?list 鏃讹紝灏嗚妯″潡鏍囪涓?backbone 骞跺惎鐢ㄧ储寮曞亸绉?=====
        if isinstance(c2, list) and m not in tuple_output_modules:
            is_backbone = True
            if not m_created:
                if n != 1:
                    raise ValueError(f"{m.__name__ if hasattr(m, '__name__') else m}: 澶氳緭鍑轰富骞蹭笉鏀寔 repeats>1")
                m_instance = m(*args)
            m_ = m_instance
            # 渚?forward/predict 璺緞璇嗗埆
            m_.backbone = True
        else:
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = (i + 4 if is_backbone else i), f, t  # attach index, 'from' index, type

        # ===== MULTIMODAL EXTENSION START - 澶氭ā鎬佸睘鎬ц缃?=====
        # Set multimodal attributes on the module if this layer has multimodal routing
        if mm_attributes and mm_router:
            mm_router.set_module_attributes(m_, mm_attributes)
        # ===== MULTIMODAL EXTENSION END =====
        if verbose:
            # Format args for display - convert class objects to simple names
            display_args = []
            for arg in args:
                if isinstance(arg, type):
                    display_args.append(arg.__name__)
                else:
                    display_args.append(arg)
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(display_args):<30}")  # print
        save_base = (i + 4) if is_backbone else i
        save.extend(x % save_base for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        # ===== HOOK EXTENSION START - 绗叚瀛楁Hook娉ㄥ唽 =====
        if MULTIMODAL_AVAILABLE and hook_manager is not None and hook_parser is not None:
            try:
                if len(layer_config) >= 6:
                    hook_field = layer_config[5]
                    specs = hook_parser.parse_hook_field(hook_field, layer_idx=i)
                    if specs:
                        if mm_input_source == 'Dual':
                            raise ValueError(
                                f"Found Dual input source at layer {i}, but contrastive hooks were also configured. "
                                "Contrastive hooks are not allowed on early Dual-fusion path."
                            )
                        for spec in specs:
                            # 娉ㄥ唽鏃舵棤闇€鏄惧紡buffer锛涜嚜鍔ㄥ懡鍚嶇敱HookManager瀹屾垚
                            hook_manager.register(m_, spec)
            except Exception as e:
                raise
        # ===== HOOK EXTENSION END =====
        if i == 0:
            ch = []
        if isinstance(c2, list) and m not in tuple_output_modules:
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)

    model = torch.nn.Sequential(*layers)
    if mm_router:
        model.multimodal_router = mm_router
    if hook_manager is not None and hook_manager.has_hooks():
        model.mm_hook_manager = hook_manager
        try:
            summary = hook_manager.summary_text(max_items=20)
            for line in summary.splitlines():
                LOGGER.info(line)
        except Exception:
            pass
    
    return model, sorted(save)


def yaml_model_load(path):
    """
    Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)  # noqa
    except AttributeError:
        return ""


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect





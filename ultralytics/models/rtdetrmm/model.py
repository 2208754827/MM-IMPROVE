# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RTDETRMM：独立的多模态 RT-DETR 家族模型入口。

设计目标：
- 不继承 RTDETR；不依赖 ultralytics.models.rtdetr.*。
- 不依赖文件名是否包含 "-mm"；而是基于 YAML/CKPT 的内容判据严格 Fail-Fast。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .mm_predictor import RTDETRMMPredictor  # 新推理引擎适配器
from .train import RTDETRMMTrainer
from .val import RTDETRMMValidator
from .utils import require_multimodal


class RTDETRMM(Model):
    """RT-DETR MultiModal（独立家族）。"""

    def __init__(
        self,
        model: Union[str, Path] = "rtdetr-r18-mm.pt",
        ch: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # Fail-Fast：必须能从内容判据识别为多模态结构
        require_multimodal(model, who="RTDETRMM")

        self.input_channels = ch
        self.modality_config: Dict[str, Any] = {}
        self.is_multimodal = True

        # 基类会触发 _new/_load，依赖 task_map
        super().__init__(model=str(model), task="detect", verbose=verbose)

        # 解析 YAML 结构并填充 modality_config；确保 mm_router 可用
        self._configure_multimodal_settings(verbose=verbose)
        self._ensure_mm_router(verbose=verbose)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": RTDETRMMPredictor,  # 新推理引擎适配器
                "validator": RTDETRMMValidator,
                "trainer": RTDETRMMTrainer,
                "model": RTDETRDetectionModel,
            }
        }

    def _configure_multimodal_settings(self, verbose: bool = False) -> None:
        """基于 self.model.yaml 识别路由标记，推断输入通道与可用模态集合。"""
        try:
            model_yaml = getattr(self.model, "yaml", None)
            if not isinstance(model_yaml, dict) or not model_yaml:
                # 无 yaml 信息时，只能使用保守配置
                self.input_channels = self.input_channels or 3
                self.modality_config = {
                    "rgb_channels": [0, 1, 2],
                    "x_channels": [3, 4, 5],
                    "supported_modalities": ["RGB"],
                    "default_modality": "RGB",
                }
                return

            has_mm = self._detect_multimodal_layers(model_yaml)
            model_channels = int(model_yaml.get("ch", model_yaml.get("channels", 3)) or 3)
            if has_mm and self._has_dual_modality_layers(model_yaml):
                model_channels = 6
            elif has_mm:
                model_channels = 3

            if self.input_channels is None:
                self.input_channels = model_channels
            self._validate_input_channels()

            if has_mm and self.input_channels == 6:
                self.modality_config = {
                    "rgb_channels": [0, 1, 2],
                    "x_channels": [3, 4, 5],
                    "supported_modalities": ["RGB", "X", "Dual"],
                    "default_modality": "Dual",
                }
            elif has_mm:
                self.modality_config = {
                    "rgb_channels": [0, 1, 2],
                    "x_channels": [3, 4, 5],
                    "supported_modalities": ["RGB", "X"],
                    "default_modality": "RGB",
                }
            else:
                self.modality_config = {
                    "rgb_channels": [0, 1, 2],
                    "x_channels": [],
                    "supported_modalities": ["RGB"],
                    "default_modality": "RGB",
                }

            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.info(
                    f"RTDETRMM configured: ch={self.input_channels}, "
                    f"modalities={self.modality_config.get('supported_modalities', [])}"
                )

        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: multimodal settings 配置失败：{e}")
            self.input_channels = self.input_channels or 3
            self.modality_config = {
                "supported_modalities": ["RGB"],
                "default_modality": "RGB",
            }

    def _detect_multimodal_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5 and str(layer[4]) in ("RGB", "X", "Dual"):
                return True
        return False

    def _has_dual_modality_layers(self, model_yaml: dict) -> bool:
        all_layers = model_yaml.get("backbone", []) + model_yaml.get("head", [])
        for layer in all_layers:
            if isinstance(layer, list) and len(layer) >= 5 and str(layer[4]) == "Dual":
                return True
        return False

    def _validate_input_channels(self) -> None:
        supported = {3, 6}
        if self.input_channels not in supported:
            raise ValueError(
                f"RTDETRMM: Unsupported input channels: {self.input_channels}. "
                f"Supported: {sorted(supported)} (3=RGB-only, 6=RGB+X)"
            )

    def _ensure_mm_router(self, verbose: bool = False) -> None:
        """确保底层 RTDETRDetectionModel 持有 mm_router。"""
        m = getattr(self, "model", None)
        if m is None:
            return

        # 直连 PyTorch 模型
        if hasattr(m, "mm_router") and m.mm_router is not None:
            return

        # AutoBackend(PyTorch)
        if hasattr(m, "pt") and getattr(m, "pt", False) and hasattr(m, "model"):
            inner = getattr(m, "model", None)
            if inner is not None and hasattr(inner, "mm_router") and inner.mm_router is not None:
                return

        # 尝试补建
        try:
            from ultralytics.nn.mm import MultiModalRouter

            config_dict = getattr(m, "yaml", None)
            if hasattr(m, "mm_router"):
                m.mm_router = MultiModalRouter(config_dict, verbose=verbose)
            elif hasattr(m, "model") and getattr(m, "pt", False):
                inner = getattr(m, "model", None)
                if inner is not None and hasattr(inner, "mm_router"):
                    inner.mm_router = MultiModalRouter(config_dict, verbose=verbose)
        except Exception as e:
            if verbose:
                from ultralytics.utils import LOGGER

                LOGGER.warning(f"RTDETRMM: 无法创建 MultiModalRouter：{e}")

    def get_modality_info(self) -> Dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "modality_config": dict(self.modality_config),
            "model_type": "RTDETRMM",
            "task": getattr(self, "task", "detect"),
            "is_multimodal": True,
        }

    def vis(
        self,
        rgb_source: Optional[Union[str, np.ndarray, list]] = None,
        x_source: Optional[Union[str, np.ndarray, list]] = None,
        method: str = "heat",
        layers: Optional[list[int]] = None,
        modality: Optional[str] = None,
        save: bool = True,
        overlay: Optional[str] = None,
        project: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        out_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """可视化入口：委托给 RTDETRMMVisualizationRunner（独立家族）。"""
        from .visualize.runner import RTDETRMMVisualizationRunner

        return RTDETRMMVisualizationRunner.run(
            model=self,
            rgb_source=rgb_source,
            x_source=x_source,
            method=method,
            layers=layers,
            modality=modality,
            save=save,
            overlay=overlay,
            project=str(project) if project is not None else None,
            name=name,
            out_dir=str(out_dir) if out_dir is not None else None,
            device=device,
            **kwargs,
        )

    def val(self, validator=None, **kwargs: Any):
        """
        多模态 RT-DETR 的验证入口。

        关键点：
        - RT-DETR 的后处理默认按 `imgsz`（单一标量）缩放 bbox，天然假设输入为方形；
        - `Model.val()` 的通用默认值是 `rect=True`，会产生非方形 batch shape，从而导致 bbox 缩放错误、mAP 异常偏低；
        - 因此这里对 RTDETRMM 默认设置 `rect=False`（用户显式传入 rect 时尊重用户设置）。
        """
        kwargs.setdefault("rect", False)
        return super().val(validator=validator, **kwargs)

    def cocoval(self, validator=None, **kwargs: Any):
        """
        使用 COCO 评估指标对 RTDETRMM 进行验证。

        说明：
        - 返回包含 `metrics/coco/*` 的指标字典（12 项 COCO 标准指标为主）；
        - 默认 `rect=False`，避免 RT-DETR 在非方形输入下因 bbox 缩放假设导致指标异常（用户显式传入 rect 时尊重用户设置）。
        """
        # COCO 验证器仅支持 PyTorch 模型
        self._check_is_pytorch_model()

        # RT-DETR 默认以方形推理为基准；若用户需要 rect 提速可显式开启
        kwargs.setdefault("rect", False)

        # COCO 验证常用默认参数
        kwargs.setdefault("save_json", True)
        kwargs.setdefault("plots", True)
        kwargs.setdefault("conf", 0.05)

        args = {**self.overrides, **kwargs, "mode": "cocoval"}

        from ultralytics.models.rtdetr.cocoval import RTDETRMMCOCOValidator

        validator = (validator or RTDETRMMCOCOValidator)(
            dataloader=None,
            save_dir=None,
            pbar=None,
            args=args,
            _callbacks=self.callbacks,
        )
        results = validator(model=self.model)
        self.metrics = results
        return results

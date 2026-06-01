#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR模型配置
统一管理模型名称、版本等配置
"""

import os

# 模型版本，可通过环境变量覆盖
MODEL_REVISION = os.environ.get("FUNASR_MODEL_REVISION", "v2.0.5")

# 模型配置（全部可通过环境变量覆盖）
#
# ASR 模型可选用两种：
#   - 标准 Paraformer: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
#   - Contextual Paraformer（支持热词编码偏置）:
#     需先通过 FunASR AutoModel 导出 ONNX 后使用，或使用预导出 ONNX 版本
#     模型名示例: iic/speech_paraformer-large-contextual-zh-cn-16k-common
#
# 设置环境变量 FUNASR_ASR_MODEL=... 可临时切换
MODELS = {
    "asr": {
        "name": os.environ.get(
            "FUNASR_ASR_MODEL",
            os.path.join(
                os.path.expanduser("~/.cache/huggingface/hub"),
                "models--JunHowie--speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
                "snapshots",
                "stable",
            ),
        ),
        "type": "asr",
    },
    "vad": {
        "name": os.environ.get(
            "FUNASR_VAD_MODEL",
            "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
        ),
        "type": "vad",
    },
    "punc": {
        "name": os.environ.get(
            "FUNASR_PUNC_MODEL",
            "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        ),
        "type": "punc",
    },
}


def is_contextual_model(model_name: str) -> bool:
    """判断模型是否是 ContextualParaformer 系列。

    ContextualParaformer/SeacoParaformer 需要 model.onnx + model_eb.onnx
    两个 ONNX 文件，通过模型名特征判断。
    """
    name_lower = model_name.lower()
    return any(kw in name_lower for kw in ["contextual", "seaco", "hotword"])


# 获取模型列表（用于下载脚本）
def get_models_for_download():
    """返回用于下载的模型配置列表"""
    return [
        {
            "name": MODELS["asr"]["name"],
            "type": "asr",
        },
        {
            "name": MODELS["vad"]["name"],
            "type": "vad",
        },
        {
            "name": MODELS["punc"]["name"],
            "type": "punc",
        },
    ]

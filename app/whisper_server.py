#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 模型服务器
兼容 FunASRServer 接口：initialize() / transcribe_audio() / cleanup()
使用 faster-whisper 本地推理。
"""

from __future__ import annotations

import logging
import os
import time
import traceback

logger = logging.getLogger(__name__)

# 模型路径映射：简体中文模型名 → HuggingFace 仓库名
MODEL_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# 默认模型（用户已有缓存）
DEFAULT_MODEL = "Systran/faster-whisper-base"


class WhisperServer:
    """Whisper 本地推理服务，接口兼容 FunASRServer。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.initialized = False

    @staticmethod
    def _ensure_cuda_lib_path() -> None:
        """将 venv 内的 CUDA 运行时库加入加载路径。

        nvidia-cublas-cu12 / nvidia-cuda-nvrtc-cu12 的 .so 文件安装在
        site-packages/nvidia/*/lib/ 下，CTranslate2 使用 ctypes/dlopen
        加载 libcublas.so.12，不走 LD_LIBRARY_PATH，需要显式预加载。
        """
        try:
            import sys
            import ctypes
            venv_site = os.path.join(
                sys.prefix, "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            )
            libs = [
                os.path.join(venv_site, "nvidia", "cublas", "lib", "libcublas.so.12"),
                os.path.join(venv_site, "nvidia", "cublas", "lib", "libcublasLt.so.12"),
            ]
            for lib in libs:
                if os.path.exists(lib):
                    ctypes.cdll.LoadLibrary(lib)
                    logger.debug("预加载 CUDA 库: %s", lib)
        except Exception as exc:
            logger.warning("CUDA 库预加载失败（非 GPU 环境可忽略）: %s", exc)

    def initialize(self) -> dict:
        """加载 Whisper 模型。首次加载会自动下载到 HF 缓存。"""
        if self.initialized:
            return {"success": True, "message": "模型已初始化"}

        # 确保 CUDA 运行时库（libcublas 等）在加载路径中
        self._ensure_cuda_lib_path()

        try:
            from faster_whisper import WhisperModel

            # resolve 简短模型名
            resolved = MODEL_MAP.get(self.model_name.lower(), self.model_name)
            logger.info(
                "正在加载 Whisper 模型: %s (resolved: %s)",
                self.model_name,
                resolved,
            )

            # 默认用 GPU（CUDA），不存在时回退 CPU
            device = os.environ.get("WHISPER_DEVICE", "cuda")
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

            self.model = WhisperModel(
                resolved,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
            self.initialized = True
            logger.info("Whisper 模型加载成功: %s (%s, %s)", resolved, device, compute_type)
            return {"success": True, "message": "模型加载成功"}
        except Exception as exc:
            error_msg = f"Whisper 模型加载失败: {exc}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "init_error"}

    def transcribe_audio(self, audio_path: str, options: dict | None = None) -> dict:
        """转录音频文件，返回格式兼容 FunASRServer。

        Args:
            audio_path: WAV 文件路径
            options: 可选参数字典，支持:
                - language: 语言代码，默认 "zh"
                - hotword: 热词（faster-whisper 不支持原生热词，忽略）
                - use_vad: 是否使用 VAD（faster-whisper 内置 VAD）

        Returns:
            dict with keys: success, text, raw_text, confidence, duration, language
        """
        if not self.initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return init_result

        if not os.path.exists(audio_path):
            return {"success": False, "error": f"音频文件不存在: {audio_path}"}

        opts = dict(options or {})
        language = opts.get("language", "zh")
        use_vad = opts.get("use_vad", True)

        try:
            duration = self._get_audio_duration(audio_path)
            logger.info(
                "开始转录音频: %s (duration=%.2fs, lang=%s, vad=%s)",
                audio_path,
                duration,
                language,
                use_vad,
            )

            start = time.time()
            segments, info = self.model.transcribe(
                audio_path,
                language=language if language else None,
                vad_filter=use_vad,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=200,
                ) if use_vad else None,
            )

            raw_parts = []
            all_segments = []
            for seg in segments:
                raw_parts.append(seg.text)
                all_segments.append(seg)

            raw_text = " ".join(raw_parts).strip()
            inference_latency = time.time() - start

            # 没有检测到语音
            if not raw_text:
                logger.info("Whisper 未识别到语音内容")
                return {
                    "success": True,
                    "text": "",
                    "raw_text": "",
                    "confidence": 0.0,
                    "duration": duration,
                    "language": str(info.language) if info and hasattr(info, "language") else language,
                    "model_type": "faster_whisper",
                    "models": {"whisper": self.model_name},
                }

            # 用平均概率作为置信度
            confidence = 0.0
            if all_segments:
                probs = [s.avg_logprob for s in all_segments if hasattr(s, "avg_logprob") and s.avg_logprob is not None]
                if probs:
                    confidence = sum(probs) / len(probs)
                    # avg_logprob 一般是负数（如 -0.3），归一化到 0-1
                    confidence = max(0.0, min(1.0, (confidence + 5.0) / 5.0))

            logger.info(
                "Whisper 转录完成: %.2fs, conf=%.3f, text=%s...",
                inference_latency,
                confidence,
                raw_text[:60],
            )

            return {
                "success": True,
                "text": raw_text,       # faster-whisper 自带标点
                "raw_text": raw_text,
                "confidence": confidence,
                "duration": duration,
                "language": str(info.language) if info and hasattr(info, "language") else language,
                "model_type": "faster_whisper",
                "models": {"whisper": self.model_name},
            }
        except Exception as exc:
            error_msg = f"Whisper 转录失败: {exc}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg, "type": "transcription_error"}

    def cleanup(self) -> None:
        """释放模型资源。"""
        if self.model:
            try:
                del self.model
                self.model = None
                self.initialized = False
                logger.info("Whisper 模型已释放")
            except Exception as exc:
                logger.warning("Whisper 模型释放时出错: %s", exc)

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        try:
            import librosa
            return librosa.get_duration(path=audio_path)
        except Exception:
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                return info.duration
            except Exception:
                return 0.0

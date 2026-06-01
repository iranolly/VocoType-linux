# VoCoType 功能增强与修改说明

本文档记录了基于上游 [LeonardNJU/VocoType-linux](https://github.com/LeonardNJU/VocoType-linux) 所做的功能增强和修改。

---

## 目录

- [1. ASR 后处理管线](#1-asr-后处理管线)
- [2. ONNX ContextualParaformer GPU 加速](#2-onnx-contextualparaformer-gpu-加速)
- [3. Whisper 引擎支持](#3-whisper-引擎支持)
- [4. Shift 键中英切换](#4-shift-键中英切换)
- [5. GPU 推理](#5-gpu-推理)
- [6. 配置文件总览](#6-配置文件总览)

---

## 1. ASR 后处理管线

**涉及文件：** `app/text_postprocessor.py`（新增）、`app/config.py`、`app/funasr_server.py`

ASR 识别结果在提交到输入框之前，会经过一个后处理管线，依次执行：

### 1.1 替换词典（ReplacementDict）

精确子串替换，用于缩写展开、术语统一。例如把 "AI" 替换为 "人工智能"。

**配置文件：** `~/.config/vocotype/replacements.json`

```json
{
  "AI": "人工智能",
  "GPU": "图形处理器"
}
```

- Key 按长度降序优先匹配（长词优先）
- 支持智能后缀重叠检测，避免重复（如 "达梦" + "数据库" + 原文已有 "库" → 不会变成 "达梦数据库库"）

### 1.2 专有名词拼音匹配（ProperNounMatcher）

滑动窗口 + 拼音模糊匹配，识别同音错字（ASR 常见错误）。支持：

| 匹配策略 | 示例 | 说明 |
|---------|------|------|
| 带调拼音精确 | 其安信 → 奇安信 | `qi2 an1 xin4` 完全一致 |
| 无调拼音匹配 | 易扬科技 → 亿阳科技 | 同音不同字 |
| 拼音近似匹配（≥3 字） | 至软科技 → 智软科技 | 允许 1 个音节不同 |
| 英文 Levenshtein | Vocotype → VoCoType | 编辑距离 ≤ len/3 |

**配置文件：** `~/.config/vocotype/proper_nouns.json`

```json
[
  "卷积神经网络",
  "吴恩达",
  "VoCoType"
]
```

### 1.3 处理顺序

```
替换词典（精确） → 中文专名（拼音匹配） → 英文专名（模糊匹配）
```

各步骤共享 occupied 标记，已替换的位置不会被后续步骤重复处理。

### 配置

在 `~/.config/vocotype/ibus.json` 中：

```json
{
  "postprocessor": {
    "enabled": true,
    "proper_nouns_path": "~/.config/vocotype/proper_nouns.json",
    "replacements_path": "~/.config/vocotype/replacements.json"
  }
}
```

---

## 2. ONNX ContextualParaformer GPU 加速

**涉及文件：** `app/funasr_config.py`、`app/funasr_server.py`

### 2.1 自动检测 ONNX 模型

`funasr_server.py` 在加载 ASR 模型时按以下优先级自动检测：

1. **本地 ONNX 量化**（最快）— `model_quant.onnx` + `model_eb_quant.onnx`
2. **本地 ONNX FP32** — `model.onnx` + `model_eb.onnx`
3. **ModelScope ONNX** — 通过 funasr_onnx 加载标准 ONNX 模型
4. **PyTorch 回退** — 使用 funasr AutoModel（GPU 或 CPU）

### 2.2 热词编码偏置

ContextualParaformer 支持热词偏置（Hotword Biasing），可在编码阶段提升指定词汇的识别准确率。

热词自动从 `~/.config/vocotype/proper_nouns.json` 加载，只传入中文词汇（英文词汇不在模型 vocab 中，由后处理管线处理）。

### 2.3 性能对比

| 模式 | 模型大小 | 推理耗时 | 备注 |
|------|---------|---------|------|
| PyTorch FP32 | ~870MB | ~487ms | AutoModel GPU |
| ONNX FP32 | ~831MB | — | 精度不变 |
| ONNX int8 量化 | ~234MB | ~282ms | **比 PyTorch 快 ~42%** |

### 2.4 模型导出

如需自行从 HuggingFace PyTorch 模型导出 ONNX：

```bash
python -c "
from funasr import AutoModel
model = AutoModel(model='JunHowie/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404')
model.export()
"
```

导出的 ONNX 文件位于 HuggingFace 缓存目录的 `model.onnx` 和 `model_eb.onnx`。

---

## 3. Whisper 引擎支持

**涉及文件：** `app/whisper_server.py`（新增）

新增基于 `faster-whisper` 的 ASR 引擎，接口与 FunASRServer 兼容（`initialize()` / `transcribe_audio()` / `cleanup()`）。可与原有 FunASR 引擎切换使用。

### 3.1 模型支持

支持 `faster-whisper` 所有模型（从 `tiny` 到 `large-v3`），默认使用 `base`：

```json
{
  "asr_engine": "whisper",
  "whisper": {
    "model": "Systran/faster-whisper-medium"
  }
}
```

### 3.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHISPER_DEVICE` | `cuda` | 计算设备（`cuda` / `cpu`） |
| `WHISPER_COMPUTE_TYPE` | `float16` | 精度（`float16` / `int8_float16`） |

### 3.3 自动 CUDA 库加载

`WhisperServer` 在初始化时会自动从 venv 的 `site-packages/nvidia/` 目录预加载 `libcublas.so.12` 等 CUDA 运行时库，确保在虚拟环境中正确使用 GPU。

---

## 4. Shift 键中英切换

**涉及文件：** `ibus/engine.py`

### 4.1 功能说明

- **点按 Shift**：切换中文/英文输入法模式
  - 中 → 英：Rime 组合被清除，后续按键直接输出英文
  - 英 → 中：重新启用 Rime 拼音
- **Shift + 其他按键**（Shift+F9、Ctrl+Shift+F9 等）：正常触发原有功能，不切换模式
- **录音期间**（F9 按住时）：Shift 操作被忽略

### 4.2 原始按键缓冲

当 Rime 处于组合状态时有原始按键缓冲机制：
- 输入拼音时，按键被缓存
- 按 Shift 切换英文模式时，缓存的原始按键会被提交（清空 Rime 组合）
- Backspace 会从缓冲中移除最后一个字符

### 4.3 辅助提示

切换模式时会在候选栏显示短暂的 `⌨ 中` / `⌨ 英` 提示（1.5 秒后自动消失）。

---

## 5. GPU 推理

**涉及文件：** `app/transcribe.py`

在导入 FunASRServer 之前设置 `FUNASR_DEVICE=cuda:0`，强制所有 ASR 推理使用 GPU CUDA 加速。

可覆盖关闭 GPU：

```bash
export FUNASR_DEVICE=cpu
ibus restart
```

### 5.1 末尾标点清理

ASR 识别结果末尾的句号、逗号会被自动移除（`.rstrip("。，.,")`），避免在聊天、搜索栏等短输入场景中产生不必要的标点。

---

## 6. 配置文件总览

### `~/.config/vocotype/ibus.json`

| 配置段 | 说明 |
|--------|------|
| `slm` | 长句润色（本地/远程 LLM） |
| `postprocessor` | ASR 后处理管线配置 |
| `whisper` | Whisper 引擎参数 |
| `asr_engine` | 选择引擎：`funasr`（默认）/ `whisper` |

### `~/.config/vocotype/proper_nouns.json`

专有名词列表（JSON 数组），同时用于：
1. ASR 热词偏置编码（中文词 → ContextualParaformer hotword）
2. 后处理拼音模糊匹配

### `~/.config/vocotype/replacements.json`

替换词典（JSON 对象），精确子串替换。

---

## 文件变更清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `app/config.py` | 修改 | 新增 `postprocessor` 配置段 |
| `app/funasr_config.py` | 修改 | 默认模型改为本地 ContextualParaformer ONNX；新增 `is_contextual_model()` |
| `app/funasr_server.py` | 修改 | 自动检测 ONNX/PyTorch；热词偏置；后处理管线集成；末尾标点清理 |
| `app/transcribe.py` | 修改 | 强制 GPU 推理（`FUNASR_DEVICE=cuda:0`） |
| `ibus/engine.py` | 修改 | Shift 点按中英切换；原始按键缓冲 |
| `app/text_postprocessor.py` | **新增** | 后处理管线：替换词典 + 专名拼音匹配 |
| `app/whisper_server.py` | **新增** | faster-whisper ASR 引擎（接口兼容 FunASRServer） |
| `docs/MODIFICATIONS.md` | **新增** | 本文档 |

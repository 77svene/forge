# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default model lists for inference, split by platform."""

import utils.hardware.hardware as hw

DEFAULT_MODELS_GGUF = [
    "forge/Llama-3.2-1B-Instruct-GGUF",
    "forge/Llama-3.2-3B-Instruct-GGUF",
    "forge/Llama-3.1-8B-Instruct-GGUF",
    "forge/gemma-3-1b-it-GGUF",
    "forge/gemma-3-4b-it-GGUF",
    "forge/Qwen3-4B-GGUF",
]

DEFAULT_MODELS_STANDARD = [
    "forge/Qwen3-4B-Instruct-2507",
    "forge/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "forge/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "forge/Phi-3.5-mini-instruct",
    "forge/Gemma-3-4B-it",
    "forge/Qwen2-VL-2B-Instruct-bnb-4bit",
]


def get_default_models() -> list[str]:
    hw.get_device()  # ensure detect_hardware() has run
    if hw.CHAT_ONLY:
        return list(DEFAULT_MODELS_GGUF)
    return list(DEFAULT_MODELS_STANDARD)

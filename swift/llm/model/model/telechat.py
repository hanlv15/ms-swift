# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model

register_model(
    ModelMeta(
        LLMModelType.telechat,
        [
            ModelGroup([
                Model('TeleAI/TeleChat-7B', 'Tele-AI/telechat-7B'),
                Model('TeleAI/TeleChat-12B', 'Tele-AI/TeleChat-12B'),
            ]),
        ],
        TemplateType.telechat,
        get_model_tokenizer_from_local,
        support_flash_attn=True,
        architectures=['TelechatForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.telechat2,
        [
            ModelGroup([
                Model('TeleAI/TeleChat2-115B', 'Tele-AI/TeleChat2-115B'),
                Model('TeleAI/TeleChat-12B-v2', 'Tele-AI/TeleChat-12B-v2'),
            ]),
            ModelGroup([
                Model('swift/TeleChat-12B-V2-GPTQ-Int4'),
            ], requires=['auto_gptq>=0.5']),
        ],
        TemplateType.telechat_v2,
        get_model_tokenizer_from_local,
        support_flash_attn=True,
        architectures=['TelechatForCausalLM'],
    ))
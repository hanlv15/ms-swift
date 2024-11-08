# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from transformers import PreTrainedTokenizerBase

from .base import Template
from .template_meta import TemplateMeta

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    TEMPLATE_MAPPING[template_type] = template_meta


def get_template(
        template_type: str,
        tokenizer: PreTrainedTokenizerBase,
        default_system: Optional[str] = None,
        max_length: Optional[int] = None,
        *,
        use_generate_template: bool = False,
        truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
        max_pixels: Optional[int] = None,  # h * w
        tools_prompt: str = 'react_en',
        # train
        loss_scale: str = 'default',
        sequence_parallel_size: int = 1) -> 'Template':
    template_meta = TEMPLATE_MAPPING[template_type]
    template_cls = template_meta.template_cls
    return template_cls(
        tokenizer,
        template_meta,
        default_system,
        max_length,
        use_generate_template=use_generate_template,
        truncation_strategy=truncation_strategy,
        loss_scale=loss_scale,
        max_pixels=max_pixels,
        sequence_parallel_size=sequence_parallel_size,
        tools_prompt=tools_prompt)


def get_template_meta(template_type: str) -> TemplateMeta:
    return TEMPLATE_MAPPING[template_type]
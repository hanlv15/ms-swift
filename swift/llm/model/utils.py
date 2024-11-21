# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
import transformers
from accelerate.utils import find_device
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from packaging import version
from transformers import AutoConfig, PretrainedConfig

from swift.hub import HFHub, MSHub, default_hub
from swift.llm import to_device
from swift.utils import deep_getattr, get_logger, is_dist, is_dist_ta, safe_ddp_context, subprocess_run

logger = get_logger()

_T = TypeVar('_T')


class AttnImpl:
    flash_attn = 'flash_attn'
    sdpa = 'sdpa'
    eager = 'eager'

    @staticmethod
    def to_use_flash_attn(attn_impl: Optional[str], auto_value: _T = None) -> Union[bool, _T]:
        if attn_impl is None:
            return auto_value
        return attn_impl == AttnImpl.flash_attn

    @staticmethod
    def update_attn_impl(config: PretrainedConfig, attn_impl: Optional[str], auto_value: _T = None) -> None:

        use_flash_attn = AttnImpl.to_use_flash_attn(attn_impl, auto_value)
        if use_flash_attn is None:
            return
        from swift.llm import HfConfigFactory
        if version.parse(transformers.__version__) >= version.parse('4.36'):
            if use_flash_attn:
                attn_impl = 'flash_attention_2'
            HfConfigFactory.set_config_attr(config, '_attn_implementation', attn_impl)
        else:
            HfConfigFactory.set_config_attr(config, '_flash_attn_2_enabled', use_flash_attn)


@dataclass
class ModelInfo:
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'aqlm', None]
    quant_bits: int

    # extra
    attn_impl: AttnImpl
    rope_scaling: Optional[Dict[str, Any]]
    config: Optional[PretrainedConfig] = None


class HfConfigFactory:
    """This class is used to read config from config.json(maybe params.json also)"""

    @staticmethod
    def get_torch_dtype(config: Union[PretrainedConfig, Dict[str, Any]],
                        quant_info: Dict[str, Any]) -> Optional[torch.dtype]:
        for key in ['torch_dtype', 'params_dtype']:
            torch_dtype = HfConfigFactory.get_config_attr(config, key)
            if torch_dtype is not None:
                break
        torch_dtype = HfConfigFactory.to_torch_dtype(torch_dtype)
        if torch_dtype is None:
            torch_dtype = quant_info.get('torch_dtype')
        return torch_dtype

    @staticmethod
    def _get_config_attrs(config: Union[PretrainedConfig, Dict[str, Any]],
                          attr_name: str) -> List[Tuple[PretrainedConfig, Any]]:
        res = []
        for key in [None, 'language_config', 'llm_config', 'text_config']:
            if key is not None:
                if isinstance(config, dict):
                    config = config.get(key)
                else:
                    config = getattr(config, key, None)
            value = deep_getattr(config, attr_name, None)
            if value is not None:
                res.append((config, value))
        return res

    @staticmethod
    def get_config_attr(config: Union[PretrainedConfig, Dict[str, Any]], attr_name: str) -> Optional[Any]:
        """Get the value of the attribute named attr_name."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name)
        if len(attrs) == 0:
            return None
        else:
            return attrs[0][1]

    @staticmethod
    def set_config_attr(config: Union[PretrainedConfig, Dict[str, Any]], attr_name: str, value: Any) -> None:
        """Set all the attr_name attributes to value."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name)
        if len(attrs) == 0:
            attrs.append((config, None))
        for config, _ in attrs:
            if isinstance(config, dict):
                config[attr_name] = value
            else:
                setattr(config, attr_name, value)

    @staticmethod
    def get_max_model_len(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[int]:
        """Get the max length supported by the model"""
        INF = int(1e9)
        max_model_len = INF

        possible_keys = [
            'seq_length',  # qwen, chatglm
            'max_position_embeddings',  # qwen1.5, llama2
            'n_positions',  # polylm, phi-2
            'model_max_length',  # baichuan2
            # others
            'seq_len',
            'max_seq_len',
            'max_sequence_length',
            'max_seq_length',
        ]
        for key in possible_keys:
            max_len_key = HfConfigFactory.get_config_attr(config, key)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == INF:
            max_model_len = None

        return max_model_len

    @staticmethod
    def compat_zero3(config: PretrainedConfig) -> None:
        value = HfConfigFactory.get_config_attr(config, 'hidden_size')
        try:
            # AttributeError: can't set attribute 'hidden_size'
            config.hidden_size = value
        except AttributeError:
            pass

    @staticmethod
    def to_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, str):
            torch_dtype = eval(f'torch.{torch_dtype}')
        return torch_dtype

    @staticmethod
    def get_quant_info(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get quant_method, quant_bits, dtype. not support hqq/eetq now, support awq/gptq/bnb/aqlm"""
        if isinstance(config, dict):
            quantization_config = config.get('quantization_config')
        else:
            quantization_config = getattr(config, 'quantization_config', None)
        if quantization_config is None:
            return
        quantization_config = dict(quantization_config)
        quant_method = quantization_config.get('quant_method')
        res = {}
        if quant_method in {'gptq', 'awq', 'aqlm'}:
            res['quant_method'] = quant_method
            res['torch_dtype'] = torch.float16
            quant_bits = quantization_config.get('bits')
            if quant_bits is not None:
                res['quant_bits'] = quant_bits
        elif quant_method == 'bitsandbytes':
            res['quant_method'] = 'bnb'
            load_in_4bit = quantization_config.get('load_in_4bit')
            load_in_8bit = quantization_config.get('load_in_8bit')
            bnb_4bit_compute_dtype = quantization_config.get('bnb_4bit_compute_dtype')
            if load_in_4bit:
                res['quant_bits'] = 4
            elif load_in_8bit:
                res['quant_bits'] = 8
            res['torch_dtype'] = HfConfigFactory.to_torch_dtype(bnb_4bit_compute_dtype)
        return res or None

    @staticmethod
    def _get_arch_mapping():
        from .register import MODEL_MAPPING
        res = {}
        for model_type, model_meta in MODEL_MAPPING.items():
            architectures = model_meta.architectures
            for arch in architectures:
                if arch not in res:
                    res[arch] = []
                res[arch].append(model_type)
        return res

    @staticmethod
    def get_matched_model_types(config: Union[PretrainedConfig, Dict[str, Any]]) -> List[str]:
        """Get possible model_type."""
        # get possible model_types based on the model architecture.
        if isinstance(config, dict):
            arch = config['architectures'][0]
        else:
            arch = config.architectures[0]

        arch_mapping = HfConfigFactory._get_arch_mapping()
        return arch_mapping.get(arch) or []


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: Optional[bool] = None,
                           ignore_file_pattern: Optional[List[str]] = None,
                           **kwargs) -> str:
    """Download model protected by DDP context

    Args:
        model_id_or_path: The model id or model path
        revision: The model revision
        download_model: Download model bin/safetensors files or not
        use_hf: use huggingface or modelscope

    Returns:
        model_dir
    """
    if not download_model:
        if ignore_file_pattern is None:
            ignore_file_pattern = []
        ignore_file_pattern += ['*.bin', '*.safetensors']
    if (is_dist() or is_dist_ta()) and not dist.is_initialized():
        # Distributed but uninitialized
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        file_path = hashlib.md5(model_id_or_path.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        context = FileLock(file_path)
    else:
        context = safe_ddp_context()
    hub = {True: HFHub, False: MSHub, None: default_hub}[use_hf]
    with context:
        if os.path.exists(model_id_or_path):
            model_dir = model_id_or_path
        else:
            if model_id_or_path[:1] in {'~', '/'}:  # startswith
                raise ValueError(f"path: '{model_id_or_path}' not found")
            model_dir = hub.download_model(model_id_or_path, revision, ignore_file_pattern, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def git_clone_github(github_url: str,
                     local_repo_name: Optional[str] = None,
                     branch: Optional[str] = None,
                     commit_hash: Optional[str] = None) -> str:
    git_cache_dir = os.path.join(get_cache_dir(), '_github')
    os.makedirs(git_cache_dir, exist_ok=True)
    if local_repo_name is None:
        github_url = github_url.rstrip('/')
        local_repo_name = github_url.rsplit('/', 1)[1]
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context():
        if not os.path.exists(local_repo_path):
            if not github_url.endswith('.git'):
                github_url = f'{github_url}.git'
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            command_str = f"git -C '{git_cache_dir}' clone '{github_url}' {local_repo_name}"
            if branch is not None:
                command += ['--branch', branch]
                command_str += f' --branch {branch}'
            logger.info(f'Run the command: `{command_str}`')
            subprocess_run(command)

            if commit_hash is not None:
                git_cache_path = os.path.join(git_cache_dir, local_repo_name)
                command = ['git', '-C', git_cache_path, 'reset', '--hard', commit_hash]
                command_str = f"git -C '{git_cache_path}' reset '--hard' {commit_hash}"
                logger.info(f'Run the command: `{command_str}`')
                subprocess_run(command)

        logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def use_submodel_func(model, submodel_name: str, func_list: List[str]) -> None:
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel.__class__, func_name)

        @wraps(_old_func)
        def _new_func(self, *args, **kwargs):
            res = _old_func(submodel, *args, **kwargs)
            if func_name == 'forward':
                device = find_device(args)
                if device is None:
                    device = find_device(kwargs)
                res.logits = to_device(res.logits, device)
                res.loss = to_device(res.loss, device)
            return res

        return _new_func

    for key in func_list:
        setattr(model, key, MethodType(_get_new_func(key), model))
        if key == 'generate' and model.device != submodel.device:
            submodel.__class__.device = model.device
        if key == 'forward' and 'generate' in func_list:
            setattr(submodel, key, MethodType(_get_new_func(key), submodel))  # fix device_map


@contextmanager
def ignore_check_imports():
    import transformers.dynamic_module_utils as td

    @wraps(td.check_imports)
    def _check_imports(filename) -> List[str]:
        return td.get_relative_imports(filename)

    td._old_check_imports = td.check_imports
    td.check_imports = _check_imports
    try:
        yield
    finally:
        td.check_imports = td._old_check_imports

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile

from swift.llm import ExportArguments, ModelMeta, SwiftPipeline
from swift.tuners import Swift
from swift.utils import get_logger
from .utils import prepare_pt_engine_template, save_checkpoint

logger = get_logger()


def merge_lora(args: ExportArguments, replace_if_exists=False, device_map=None) -> None:
    if replace_if_exists:
        logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.ckpt_dir is not None, f'args.ckpt_dir: {args.ckpt_dir}'
    assert args.weight_type == 'adapter', f'args.weight_type: {args.weight_type}'
    assert args.quant_method is None, (f'args.quant_method: {args.quant_method}, '
                                       'quantized model and does not support merge-lora.')

    output_dir = getattr(args, 'output_dir', None)
    if output_dir is None:
        output_dir = f'{args.ckpt_dir}-merged'
    if os.path.exists(output_dir) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {output_dir}, '
                    'skipping the saving process. '
                    'you can pass `replace_if_exists=True` to overwrite it.')
    else:
        origin_device_map = args.device_map
        args.device_map = device_map or args.device_map
        logger.info(f'merge_device_map: {device_map}')
        if args.use_merge_kit:
            base_model_id_or_path = args.model_id_or_path
            if not os.path.exists(args.instruct_model_id_or_path):
                args.instruct_model_id_or_path = args.hub.download_model(
                    args.instruct_model_id_or_path, revision=args.instruct_model_revision)
            args.model_id_or_path = args.instruct_model_id_or_path
        model, template = prepare_pt_engine_template(args)
        logger.info('Merge LoRA...')
        Swift.merge_and_unload(model)
        model = model.model
        logger.info('Saving merged weights...')

        save_checkpoint(
            model,
            template.processor,
            output_dir,
            safe_serialization=args.safe_serialization,
            model_dirs=[args.ckpt_dir],
            max_shard_size=args.max_shard_size,
            additional_saved_files=model.model_meta.additional_saved_files)
        logger.info(f'Successfully merged LoRA and saved in {output_dir}.')
        args.device_map = origin_device_map

    args.weight_type = 'full'
    args.ckpt_dir = output_dir
    if args.use_merge_kit:
        tempdir = tempfile.gettempdir()
        mergekit_path = os.path.join(output_dir, 'mergekit')
        merge_yaml = args.merge_yaml.replace('{merged_model}', output_dir).replace(
            '{instruct_model}', args.instruct_model_id_or_path).replace('{base_model}', base_model_id_or_path)
        try:
            yamlfile = os.path.join(tempdir, 'mergekit.yaml')
            with open(yamlfile, 'w') as f:
                f.write(merge_yaml)
            os.system(f'mergekit-yaml {yamlfile} {mergekit_path}')
        finally:
            if tempdir:
                shutil.rmtree(tempdir, ignore_errors=True)

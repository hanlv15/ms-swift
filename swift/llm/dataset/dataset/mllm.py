import ast
import os
from typing import Any, Dict, Optional

import json
import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm

from ..media import MediaResource
from ..preprocess import MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor
from ..preprocess.extra import GroundingMixin
from ..register import DatasetMeta, SubsetDataset, register_dataset


class ShareGPT4oPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = super().preprocess(row)
        image = row['images']
        if not image:
            return
        image = os.path.join(self.prefix_path, image)
        if not os.path.exists(image):
            return
        row['images'] = [image]
        return row

    def prepare_dataset(self, dataset):
        url = ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?'
               'Revision=master&FilePath=images.zip')
        local_dir = MediaResource.download(url, 'sharegpt_4o_images')
        self.prefix_path = os.path.join(local_dir, 'mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')
        return super().prepare_dataset(dataset)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ShareGPT-4o',
        hf_dataset_id='OpenGVLab/ShareGPT-4o',
        preprocess_func=ShareGPT4oPreprocessor(columns_mapping={'image': 'images'}),
        subsets=['image_caption'],
        split=['images'],
        tags=['vqa', 'multi-modal'],
    ))


class GPT4vDataset(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'What is the caption of this image?'
        return super().preprocess(row)


# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='swift/gpt4v-dataset',
#         hf_dataset_id='laion/gpt4v-dataset',
#         preprocess_func=GPT4vDataset(columns_mapping={
#             'link': 'images',
#             'caption': 'response'
#         }),
#         subsets=['default'],
#         split=['train'],
#         tags=['en', 'caption', 'multi-modal', 'quality'],
#     ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/RLAIF-V-Dataset',
        hf_dataset_id='openbmb/RLAIF-V-Dataset',
        preprocess_func=ResponsePreprocessor(columns_mapping={
            'image': 'images',
            'question': 'query',
            'chosen': 'response',
            'rejected': 'rejected_response'
        }),
        tags=['rlhf', 'dpo', 'multi-modal', 'en'],
    ))


class SA1BPairedCaptionPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']
        response = row['global_caption']
        query = np.random.choice(prompt)
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='Tongyi-DataEngine/SA1B-Paired-Captions-Images',
        preprocess_func=SA1BPairedCaptionPreprocessor(columns_mapping={
            'opensource_url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],
    ))


class SA1BDenseCaptionPreprocessor(RowPreprocessor):

    column_mapping = {
        'url': 'images',
    }

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']
        response = ast.literal_eval(row['cap_seg'])
        response = response.get('global_caption')
        query = np.random.choice(prompt)
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='Tongyi-DataEngine/SA1B-Dense-Caption',
        preprocess_func=SA1BDenseCaptionPreprocessor(columns_mapping={
            'url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],
        huge_dataset=True,
    ))


class COCO2014Preprocess(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption = row['caption']
        if '&&' in caption:
            caption = caption.split('&&')[0]
        row['query'] = 'please describe the image.'
        row['response'] = caption
        row['images'] = row['image']['path']

        return super().preprocess(row)

    def __call__(self, dataset, **kwargs):
        from datasets import Image
        dataset = dataset.cast_column('image', Image(decode=False))
        return super().__call__(dataset, **kwargs)


register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/coco_2014_caption',
        preprocess_func=COCO2014Preprocess(),
        subsets=[
            SubsetDataset('train', 'coco_2014_caption', ['train']),
            SubsetDataset('val', 'coco_2014_caption', ['validation']),
        ],
        tags=['chat', 'multi-modal', 'vision', '🔥'],
    ))


class MantisPreprocessor(MessagesPreprocessor):

    def __init__(self,
                 *,
                 subset: str,
                 columns_mapping: Optional[Dict[str, str]] = None,
                 remove_useless_columns: bool = True) -> None:
        self.subset = subset
        super().__init__(columns_mapping=columns_mapping, remove_useless_columns=remove_useless_columns)

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        url = (f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision='
               f'master&FilePath={self.subset}/train_images.zip')  # noqa
        self.local_dir = MediaResource.download(url, f'mantis_{self.subset}')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        images = [os.path.join(self.local_dir, p['path']) for p in row['images']]
        if not all([os.path.exists(d) for d in images]):
            images = []

        if not images:
            return
        row['images'] = images
        return super().preprocess(row)


mantis_subsets_name = [
    'birds-to-words', 'chartqa', 'coinstruct', 'contrastive_caption', 'docvqa', 'dreamsim', 'dvqa', 'iconqa',
    'imagecode', 'llava_665k_multi', 'lrv_multi', 'multi_vqa', 'nextqa', 'nlvr2', 'spot-the-diff', 'star',
    'visual_story_telling'
]

_mantis_subsets = []
for subset in mantis_subsets_name:
    _subset = SubsetDataset(
        name=subset, split=['train'], preprocess_func=MantisPreprocessor(subset=subset))
    _mantis_subsets.append(_subset)

# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='swift/Mantis-Instruct',
#         subsets=_mantis_subsets,
#         tags=['chat', 'multi-modal', 'vision'],
#     ))


class LLaVADataPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row['images']:
            return
        row = super().preprocess(row)
        images = [p['path'] for p in row['images']]
        new_images = []
        for image in images:
            if 'coco/' in image:
                image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
            elif 'gqa/' in image:
                image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
            elif 'ocr_vqa/' in image:
                image = os.path.join(self.all_folders['ocr_vqa'], image)
            elif 'textvqa/' in image:
                image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
            elif 'VG_100K/' in image:
                image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
            elif 'VG_100K_2/' in image:
                image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
            new_images.append(image)
        if all(os.path.exists(image) for image in new_images):
            row['images'] = new_images
        else:
            return {'images': None}
        return row


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/llava-data',
        hf_dataset_id='TIGER-Lab/llava-data',
        subsets=['llava_instruct'],
        preprocess_func=LLaVADataPreprocessor(),
        tags=['sft', 'multi-modal', 'quality'],
    ))


class PixelProsePreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption_prompt = [
            'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
        ]
        vlm_caption = row['vlm_caption']
        if vlm_caption.startswith('This image displays:'):
            vlm_caption = vlm_caption[len('This image displays:'):].strip()
        return {
            'messages': [{
                'role': 'user',
                'content': np.random.choice(caption_prompt)
            }, {
                'role': 'assistant',
                'content': vlm_caption
            }],
            'images':
            row['url']
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/pixelprose',
        hf_dataset_id='tomg-group-umd/pixelprose',
        preprocess_func=PixelProsePreprocessor(),
        split=['train', 'cc12m', 'commonpool', 'redcaps'],
        tags=['caption', 'multi-modal', 'vision'],
        huge_dataset=True,
    ))


class AIShell1Preprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = '语音转文本'
        audio_key = 'Audio:FILE'
        response_key = 'Text:LABEL'
        return {
            'messages': [{
                'role': 'user',
                'content': prompt,
            }, {
                'role': 'assistant',
                'content': row[response_key].replace(' ', '')
            }],
            'audios':
            row[audio_key],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='speech_asr/speech_asr_aishell1_trainsets',
        preprocess_func=AIShell1Preprocessor(),
        split=['train', 'validation', 'test'],
        tags=['chat', 'multi-modal', 'audio'],
    ))


class VideoChatGPTPreprocessor(RowPreprocessor):

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        url = 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'
        local_dir = MediaResource.download(url, 'video_chatgpt')
        self.local_dir = os.path.join(local_dir, 'Test_Videos')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # only `.mp4`
        mp4_set = [file[:-4] for file in os.listdir(self.local_dir) if file.endswith('mp4')]
        if row['videos'] not in mp4_set:
            return
        return {
            'messages': [{
                'role': 'user',
                'content': row.get('question') or row.get('question_1') or rowrow.get('question_2')
            }, {
                'role': 'assistant',
                'content': row['answer']
            }],
            'videos': os.path.join(self.local_dir, f"{row['videos']}.mp4"),
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/VideoChatGPT',
        hf_dataset_id='lmms-lab/VideoChatGPT',
        subsets=['Generic', 'Temporal', 'Consistency'],
        preprocess_func=VideoChatGPTPreprocessor(columns_mapping={'video_name': 'videos'}),
        split=['test'],
        tags=['chat', 'multi-modal', 'video', '🔥'],
    ))


def preprocess_mind2web(dataset, **kwargs):

    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw_html = row['cleaned_html']
        screenshot = row['screenshot']
        row['screenshot'] = MediaResource.safe_save(screenshot, row['action_uid'] + '.jpg', 'mind2web')
        action = row['target_action_reprs']
        actions = action.split('->')
        row['query'] = f'The snapshot of screen:<image>\nThe html source code:{raw_html}\n'
        action = actions[-1]
        where = actions[0] if len(actions) > 1 else ''
        what = ''
        if ':' in action:
            action, what = action[:action.find(':')], action[action.find(':') + 1:]
        row['response'] = f'Action: {action.strip()}\nAction Input: {where.strip()}{"," + what.strip()}'
        return row

    conversations = []
    tools = [{
        'function': {
            'name': 'CLICK',
            'desc': 'Choose and click an element in the web page',
            'parameter': [{
                'element': 'string, the element in the web page to click'
            }]
        }
    }, {
        'function': {
            'name':
            'TYPE',
            'desc':
            'Input some text into a web element like <input> or <textbox>',
            'parameter': [{
                'element': 'string, the element in the web page to input to',
                'content': 'string, what content to input into the textbox elment'
            }]
        }
    }, {
        'function': {
            'name':
            'SELECT',
            'desc':
            'Select an element from a combobox',
            'parameter': [{
                'element': 'string, the combobox or dropdown in the web page on which the select happens',
                'content': 'string, which choices to choose'
            }]
        }
    }]

    def history_to_messages(history):
        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        return messages

    if isinstance(dataset, HfIterableDataset):

        def generate_example(dataset):
            history = []
            images = []
            for row in dataset:
                target_action_index = row['target_action_index']
                row = preprocess_row(row)
                query = row['query']
                if target_action_index == '0':
                    if history:
                        yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}
                        images = []
                        history = []
                    query = query + '\n' + row['confirmed_task']
                history.append([query, row['response']])
                images.append(row['screenshot'])

            if history:
                yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    history = []
    images = []
    for row in tqdm(dataset):
        target_action_index = row['target_action_index']
        row = preprocess_row(row)
        query = row['query']
        if target_action_index == '0':
            if history:
                conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append(row['screenshot'])

    if history:
        conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})

    return HfDataset.from_list(conversations)


# register_dataset(
#     DatasetMeta(
#         ms_dataset_id='swift/Multimodal-Mind2Web',
#         hf_dataset_id='osunlp/Multimodal-Mind2Web',
#         preprocess_func=preprocess_mind2web,
#         tags=['agent', 'multi-modal']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/M3IT',
        subsets=[
            'coco', 'vqa-v2', 'shapes', 'shapes-rephrased', 'coco-goi-rephrased', 'snli-ve', 'snli-ve-rephrased',
            'okvqa', 'a-okvqa', 'viquae', 'textcap', 'docvqa', 'science-qa', 'imagenet', 'imagenet-open-ended',
            'imagenet-rephrased', 'coco-goi', 'clevr', 'clevr-rephrased', 'nlvr', 'coco-itm', 'coco-itm-rephrased',
            'vsr', 'vsr-rephrased', 'mocheg', 'mocheg-rephrased', 'coco-text', 'fm-iqa', 'activitynet-qa', 'msrvtt',
            'ss', 'coco-cn', 'refcoco', 'refcoco-rephrased', 'multi30k', 'image-paragraph-captioning', 'visual-dialog',
            'visual-dialog-rephrased', 'iqa', 'vcr', 'visual-mrc', 'ivqa', 'msrvtt-qa', 'msvd-qa', 'gqa', 'text-vqa',
            'ocr-vqa', 'st-vqa', 'flickr8k-cn'
        ],
        preprocess_func=ResponsePreprocessor(columns_mapping={
            'instruction': 'system',
            'inputs': 'query',
            'image_base64_str': 'images',
            'outputs': 'response'
        }),
        split=['train'],
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class ShareGPT4VPreprocessor(RowPreprocessor):

    def prepare_dataset(self, dataset):
        split = ['ShareGPT4V', 'ShareGPT4V-PT'] if dataset.config_name is None else dataset.config_name
        IMAGE_DATASET_REQUIREMENTS = {
            'ShareGPT4V': ['coco', 'sam', 'llava', 'wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark'],
            'ShareGPT4V-PT': ['coco', 'sam', 'llava']
        }

        if isinstance(split, str):
            split = [split]
        self.all_folders = {}
        for sp in split:
            for media_type in IMAGE_DATASET_REQUIREMENTS[sp]:
                self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image = row['image']
        row.update(MessagesPreprocessor().preprocess(row))
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'sam/' in image:
            image = os.path.join(self.all_folders['sam'], image.replace('sam/images/', ''))
        elif 'llava/' in image:
            image = os.path.join(self.all_folders['llava'], image.replace('llava/llava_pretrain/images/', ''))
        elif 'wikiart/' in image:
            image = os.path.join(self.all_folders['wikiart'], image.replace('wikiart/images/', 'data/wikiart/images/'))
        elif 'share_textvqa/' in image:
            image = os.path.join(self.all_folders['share_textvqa'],
                                 image.replace('share_textvqa/images/', 'data/share_textvqa/images/'))
        elif 'web-celebrity/' in image:
            image = os.path.join(self.all_folders['web-celebrity'],
                                 image.replace('web-celebrity/images/', 'data/web-celebrity/images/'))
        elif 'web-landmark/' in image:
            image = os.path.join(self.all_folders['web-landmark'],
                                 image.replace('web-landmark/images/', 'data/web-landmark/images/'))
        if os.path.exists(image):
            row['images'] = image
        else:
            return
        return row


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ShareGPT4V',
        preprocess_func=ShareGPT4VPreprocessor(),
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class TextCapsPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = row['image']
            query = 'What is the caption of this image?'
            response = row['reference_strs']
            return {
                'messages': [
                    {
                        'role': 'user',
                        'content': query
                    },
                    {
                        'role': 'assistant',
                        'content': response[np.random.choice(range(len(response)))]
                    },
                ],
                'image':
                image
            }
        except Exception:
            return


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/TextCaps',
        hf_dataset_id='HuggingFaceM4/TextCaps',
        preprocess_func=TextCapsPreprocessor(),
        split=['train', 'validation'],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'caption', 'quality']))


class RefCOCOPreprocessor(ResponsePreprocessor, GroundingMixin):
    task_type = 'caption'

    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        super().__init__(**kwargs)

    def prepare_dataset(self, dataset):
        self.cache_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
            'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption = row['captions'][0]
        bbox = row['bbox']
        image_path = os.path.join(self.cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        if not os.path.exists(image_path):
            return

        for i in range(len(bbox)):
            bbox[i] = round(float(bbox[i]))
        res = {}

        objects = [{
            'caption': caption,
            'bbox': bbox,
            'bbox_type': 'real',
            'image': 0,
        }]
        res['query'], res['response'] = self.construct_grounding_prompt()
        res['images'] = [image_path]
        res['objects'] = objects
        return super().preprocess(res)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcoco',
        hf_dataset_id='jxu124/refcoco',
        subsets=[
            SubsetDataset(
                name='caption',
                subset='default',
                preprocess_func=RefCOCOPreprocessor('caption'),
                split=['train', 'validation'],
            ),
            SubsetDataset(
                name='grounding',
                subset='default',
                preprocess_func=RefCOCOPreprocessor('grounding'),
                split=['train', 'validation'],
            )
        ],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcocog',
        hf_dataset_id='jxu124/refcocog',
        subsets=[
            SubsetDataset(
                name='caption',
                subset='default',
                preprocess_func=RefCOCOPreprocessor('caption'),
                split=['train', 'validation'],
            ),
            SubsetDataset(
                name='grounding',
                subset='default',
                preprocess_func=RefCOCOPreprocessor('grounding'),
                split=['train', 'validation'],
            )
        ],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/lnqa',
        hf_dataset_id='vikhyatk/lnqa',
        preprocess_func=MessagesPreprocessor(user_role='question', assistant_role='answer'),
        split=['train', 'validation'],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'ocr-vqa', 'quality']))


class LLaVAInstructPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image = row['images']
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'gqa/' in image:
            image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
        elif 'ocr_vqa/' in image:
            image = os.path.join(self.all_folders['ocr_vqa'], image)
        elif 'textvqa/' in image:
            image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
        elif 'VG_100K/' in image:
            image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
        elif 'VG_100K_2/' in image:
            image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
        if os.path.exists(image):
            row['images'] = image
        else:
            return None

        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Instruct-150K',
        ms_revision='d5db3806e395c60496630a206c336932e85a2d00',
        preprocess_func=LLaVAInstructPreprocessor(columns_mapping={'image': 'images'}),
        split=['train'],
        tags=['chat', 'multi-modal', 'vision']))


class LLaVAPretrainPreprocessor(RowPreprocessor):

    def prepare_dataset(self, dataset):
        self.media_dir = MediaResource.download(
            ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?'
             'Revision=master&FilePath=images.zip'),
            # noqa
            'llava_pretrain')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row.update(MessagesPreprocessor().preprocess(row))
        if row['image']:
            file_path = os.path.join(self.media_dir, row['image'])
            if os.path.exists(file_path):
                return {'images': file_path}
            else:
                return
        else:
            return


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Pretrain',
        ms_revision='e3a3f0bfaad05e90e46745152a32bf944e0f4a63',
        hf_dataset_id='liuhaotian/LLaVA-Pretrain',
        preprocess_func=LLaVAPretrainPreprocessor(),
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/MideficsDataset',
        hf_dataset_id='WinterSchool/MideficsDataset',
        preprocess_func=MessagesPreprocessor(columns_mapping={'image': 'images'}, inner_key='data', user_role='question', assistant_role='answer'),
        tags=['medical', 'en', 'vqa']))


class OkvqaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = np.random.choice(row['answers'])
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/OK-VQA_train',
        hf_dataset_id='Multimodal-Fatima/OK-VQA_train',
        preprocess_func=OkvqaPreprocessor(columns_mapping={'image': 'images'}),
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class AOkvqaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = np.random.choice(row['rationales'])
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/A-OKVQA',
        hf_dataset_id='HuggingFaceM4/A-OKVQA',
        split=['train', 'validation'],
        preprocess_func=AOkvqaPreprocessor(columns_mapping={'image': 'images'}),
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class OcrvqaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        idx = np.random.choice(range(len(row['questions'])))
        query = row['questions'][idx]
        response = row['answers'][idx]
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/OCR-VQA',
        hf_dataset_id='howard-hou/OCR-VQA',
        split=['train', 'validation'],
        preprocess_func=OcrvqaPreprocessor(columns_mapping={'image': 'images'}),
        tags=['multi-modal', 'en', 'ocr-vqa']))


class ScienceQAPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = row['choices'][row['answer']]
        solution = row['solution']
        response = f'{solution}\nSo the final answer is: {response}'
        return {'messages': [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}]}


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/ScienceQA',
        hf_dataset_id='derek-thomas/ScienceQA',
        split=['train', 'validation'],
        preprocess_func=ScienceQAPreprocessor(columns_mapping={'image': 'images'}),
        tags=['multi-modal', 'science', 'vqa', 'quality']))


class GritPreprocessor(RowPreprocessor, GroundingMixin):

    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        super().__init__(**kwargs)

    @staticmethod
    def has_overlap(start_ends):
        for i in range(1, len(start_ends)):
            if start_ends[i][0] < start_ends[i - 1][1]:
                return True
        return False

    @staticmethod
    def replace_intervals_with_tags(response, start_ends):
        result = []
        last_end = 0
        for start, end in start_ends:
            result.append(response[int(last_end):int(start)])
            result.append('<ref-object><bbox>')
            last_end = end
        result.append(response[int(last_end):])
        return ''.join(result)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        images = row['url']
        caption = row['caption']
        ref_exps = row['ref_exps']
        objects = []
        start_end_pairs = []
        for ref_exp in ref_exps:
            start = ref_exp[0]
            end = ref_exp[1]
            # conf = ref_exp[6] TODO filter low confidence rows?
            start_end_pairs.append(ref_exp[0:2])

            object_part = caption[int(start):int(end)]
            objects.append({'caption': object_part, 'bbox': ref_exp[2:6], 'bbox_type': 'real', 'image': 0})

        start_end_pairs.sort(key=lambda x: (x[0], x[1]))
        if self.has_overlap(start_end_pairs) or not objects:
            return

        if self.task_type in ('grounding', 'caption'):
            query, response = self.construct_grounding_prompt()
        else:
            query = 'what is the proper caption of this image?'
            response = caption
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
            'images': images,
            'objects': objects
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/GRIT',
        hf_dataset_id='zzliang/GRIT',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=GritPreprocessor('caption', columns_mapping={'url': 'images'}),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=GritPreprocessor('grounding', columns_mapping={'url': 'images'}),
            ),
            SubsetDataset(
                name='vqa',
                preprocess_func=GritPreprocessor('vqa', columns_mapping={'url': 'images'}),
            )
        ],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'caption-grounding', 'vqa', 'quality']))


class GQAPreprocessor(RowPreprocessor):

    def prepare_dataset(self, dataset):
        self.local_cache = MediaResource.download('gqa')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'):
            return {
                'messages': [{
                    'role': 'user',
                    'content': row['question']
                }, {
                    'role': 'assistant',
                    'content': row['fullAnswer']
                }],
                'images':
                os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'),
            }
        else:
            return


register_dataset(
    DatasetMeta(
        hf_dataset_id='lmms-lab/GQA',
        split=['train_all_instructions'],
        preprocess_func=GQAPreprocessor(),
        huge_dataset=True,
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class LLaVAMixSFTPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row['messages']
        rounds = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text = ''
            for index in content:
                if index['type'] == 'text':
                    text += index['text']
                elif index['type'] == 'image':
                    text += '<image>'

            rounds.append({'role': role, 'content': text})

        return {'messages': rounds}


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/llava-instruct-mix-vsft',
        hf_dataset_id='HuggingFaceH4/llava-instruct-mix-vsft',
        split=['test'],
        preprocess_func=LLaVAMixSFTPreprocessor(),
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class LatexocrPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [{
                'role': 'user',
                'content': 'Using LaTeX to perform OCR on the image.'
            }, {
                'role': 'assistant',
                'content': row['text']
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LaTeX_OCR',
        hf_dataset_id='linxy/LaTeX_OCR',
        subsets=[
            SubsetDataset(
                name='default',
                split=['train'],
                preprocess_func=LatexocrPreprocessor(columns_mapping={'image': 'images'}),
            ),
            SubsetDataset(
                name='synthetic_handwrite',
                split=['train', 'validation'],
                preprocess_func=LatexocrPreprocessor(columns_mapping={'image': 'images'}),
            )
        ],
        tags=['chat', 'ocr', 'multi-modal', 'vision']))


class CapchaImagesPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = 'recognize the content.'
        response_key = 'solution'
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': row[response_key]
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/captcha-images',
        split=['train', 'validation'],
        preprocess_func=CapchaImagesPreprocessor(columns_mapping={'image': 'images'}),
        tags=['chat', 'multi-modal', 'vision']))

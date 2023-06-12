# Contrastive-Neural-Korean-Text-Generation

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers|4.21.1-pink?color=FF33CC)](https://github.com/huggingface/transformers)

## Warmup Stage
> **Note** <br>
> Empty

## Training
> **Note** <br>
> Empty

## Inference
> **Note** <br>
> Empty

## News Summarization Performance (F1-score)
After restoring the model's tokenized output to the original text, Rouge performance was evaluated by comparing it to the reference and hypothesis tokenized using [mecab](https://konlpy.org/ko/v0.4.0/).

- Dacon 한국어 문서 생성요약 AI 경진대회 [Dataset](https://dacon.io/competitions/official/235673/overview/description)
    - Training: 29,432
    - Validation: 7,358
    - Test: 9,182

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| KoBART | 124M | - | - | - |
| T5-small | 77M | - | - | - |
|  |  |  |  |  |
| CoNKT-T5-small | 77M | - | - | - |

- AI-Hub 문서요약 텍스트 [Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
    - Training: 245,626
    - Validation: 20,296
    - Test: 9,931

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| KoBART | 124M | - | - | - |
| T5-small | 77M | - | - | - |
|  |  |  |  |  |
| CoNKT-T5-small | 77M | - | - | - |

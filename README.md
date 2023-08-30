# Contrastive-Neural-Korean-Text-Generation
CoNT is a strong contrastive learning framework for neural text generation which outperforms the MLE based training method on five generation tasks, including machine translation, summarization, code comment generation, data-to-text generation, commensense generation. <br>
   - [CoNT-[NeurIPS 2022]](https://arxiv.org/abs/2205.14690) <br>
   - [[Github]](https://github.com/Shark-NLP/CoNT) Official implementation of CoNT <br>
<img src=https://github.com/BM-K/CoNKT/assets/55969260/c0613709-d797-48cf-9acb-77b5fcec8389>

The aforementioned repository has the following issues:
- 1. It does not support Korean models and BART language model
- 2. For Korean, it is not appropriate to use beam search decoding when sampling negative samples

Therefore, we release the <strong>CoNKT</strong> (Contrastive Neural Korean Text Generation) model, which solves the two issues mentioned above.

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers|4.21.1-pink?color=FF33CC)](https://github.com/huggingface/transformers)

## Warm-up Stage
A warm-up stage where the model is only supervised by `Negative Log Likelihood Loss` is recommended as it guarantees the quality of the examples from the model’s prediction.
```
# Warm-up Training & Inference (T5)
bash run_warmup_t5.sh

# Warm-up Training & Inference (BART)
bash run_warmup_bart.sh
```

## CoNKT Stage
We implement its InfoNCE version by treating ground truth as positive sample and self-generated samples (by top-p decoding strategy not beam search decoding) are also treated as negative samples.
```
# CoNKT Training & Inference (T5)
bash run_conkt_t5.sh

# CoNKT Training & Inference (BART)
bash run_conkt_bart.sh
```

## News Summarization Performance (F1-score)
After restoring the model's tokenized output to the original text, Rouge performance was evaluated by comparing it to the reference and hypothesis tokenized using [mecab](https://konlpy.org/ko/v0.4.0/).

- Dacon, Korean Abstract Summarization AI Contest [[Dataset]](https://dacon.io/competitions/official/235673/overview/description)
    - Training: 29,432
    - Validation: 7,358
    - Test: 9,182

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| T5-small | 77M | 51.55 | 33.26 | 45.02 |
| KoBART | 124M | 53.75 | 34.40 | 45.94 |
|  |  |  |  |  |
| CoNKT-T5-small | 77M | 54.08 | 34.42 | 45.54 |
| CoNKT-KoBART | 124M | 55.02 | 35.22 | 46.22 |

- AI-Hub, News Summarization [[Dataset]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
    - Training: 245,626
    - Validation: 27,685
    - Test: 2,542
 
      
| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| T5-small | 77M | 53.44 | 34.03 | 45.36 |
| KoBART | 124M | 56.04 | 36.29 | 47.15 |
|  |  |  |  |  |
| CoNKT-T5-small | 77M | 56.40 | 36.35 | 46.90 |
| CoNKT-KoBART | 124M | 58.20 | 38.95 | 49.04 |

- [KoBART](https://github.com/SKT-AI/KoBART)
- [T5-small](https://github.com/paust-team/pko-t5)

### Citing
```
@article{an2022cont,
  title={CoNT: Contrastive Neural Text Generation},
  author={An, Chenxin and Feng, Jiangtao and Lv, Kai and Kong, Lingpeng and Qiu, Xipeng and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2205.14690},
  year={2022}
}
```

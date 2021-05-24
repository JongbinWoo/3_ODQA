<div align="center">

# Rag

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[Paper](https://arxiv.org/abs/2005.11401)
</div>

## Description
- [huggingface rag](https://huggingface.co/transformers/model_doc/rag.html)모델을 수정하여 한국어 모델에서 학습할 수 있도록 작성
- dense retriever와 generator based question answering을 end-to-end방식으로 fine-tuning
- electra기반 dense retriever를 위해 model_rag.py 수정 필요
- 데이터 셋의 크기, 특성으로 인해 학습이 진행되지 않음 -> NQ 데이터셋으로 학습 권장
## How to run

Train model with default configuration
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

<br>

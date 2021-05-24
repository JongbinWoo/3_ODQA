<div align="center">

# Dense Retriever

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
- [koelectra](https://github.com/monologg/KoELECTRA)를 fine-tuning하여 question encoder와 context encoder를 학습.
- in-batch negative.(batch size: 96, V100기준)
- sampler를 활용해서 batch 구성
- huggingface datasets를 이용한 데이터 전처리 및 데이터셋 구성



```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1

python run.py experiment=example_simple
```

<br>

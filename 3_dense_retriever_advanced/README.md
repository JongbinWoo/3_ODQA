<div align="center">

# Dense Retriever Advanced

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

## Description
- [pytorch lightning moco구현](https://github.com/PyTorchLightning/lightning-bolts/tree/master/pl_bolts/models/self_supervised/moco)을 기반으로 작성
- 2에서 학습한 context encoder를 고정한 상태로 question encoder만 추가로 학습
- 전체 document embedding을 저장해놓은 상태에서 학습함으로서 question embedding을 학습할때 negative sample의 크기를 최대로 키웠다.


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

<div align="center">

# 3_ODQA
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

- 이 repo는 p stage 3주차에 진행했던 open domain question answering 코드입니다.
- pytorch lightning으로 작성되었고 huggingface transformers를 사용했습니다.
- [template](https://github.com/ashleve/lightning-hydra-template) 을 사용했습니다.


```
.
├── /1_generator_qa
│     └── generator-based question answering. kobart를 fine-tuning
├── /2_dense_retriever
│     └── in-batch negative로 학습. koelectra-small-v3를 fine-tuning
├── /3_dense_retriever_advanced
│     └── MOCO에서 아이디어를 얻어서 전체 document에 대한 question encoder 학습 진행
└── /4_rag
      └── huggingface의 예제를 참고해서 작성한 rag fine-tuning. 위의 폴더에서 학습한 모델을 사용
``` 
from typing import List, Optional

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

import hydra
from omegaconf import DictConfig

from src.utils import template_utils
from src.models.retriever_model import DualEncoder
log = template_utils.get_logger(__name__)
import torch
from tqdm import tqdm, trange

def evaluate(config: DictConfig) -> Optional[float]:
    if "seed" in config:
        seed_everything(config.seed)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning callbacks
    

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))


    valid_loader = datamodule.test_dataloader()
    CKPT_PATH = "/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-13/18-28-04/checkpoints/last.ckpt"
    model = DualEncoder.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    q_encoder = model.q_encoder
    c_encoder = model.c_encoder
    
    q_embs = []
    c_embs = []
    c_indexes = []
    c_encoder.to("cuda")
    q_encoder.to("cuda")
    with torch.no_grad():

        epoch_iterator = tqdm(
            valid_loader, desc="Iteration", position=0, leave=True
        )
        c_encoder.eval()
        q_encoder.eval()

        for _, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)
            q_inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            c_inputs = {
                "input_ids": batch[3],
                "attention_mask": batch[4],
                "token_type_ids": batch[5],
            }
            q_outputs = q_encoder(**q_inputs).to("cpu").numpy()
            c_outputs = c_encoder(**c_inputs).to("cpu").numpy()

            q_embs.extend(q_outputs)
            c_embs.extend(c_outputs)
        torch.cuda.empty_cache()
    
        # if torch.cuda.is_available():
        c_embs_cuda = torch.Tensor(c_embs)#.to("cuda")
        q_embs_cuda = torch.Tensor(q_embs)#.to("cuda")
        dot_prod_scores = torch.matmul(q_embs_cuda, torch.transpose(c_embs_cuda, 0, 1))

        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        
        correct_5 = 0
        correct_20 = 0
        correct_50 = 0
        for i, r in enumerate(rank):
            if i in r[:5]:
                correct_5 += 1
            if i in r[:20]:
                correct_20 += 1
            if i in r[:50]:
                correct_50 += 1
        print(f'top-5 Acc: {correct_5 / len(rank)}')
        print(f'top-20 Acc: {correct_20 / len(rank)}')
        print(f'top-50 Acc: {correct_50 / len(rank)}')


    

    
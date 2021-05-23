from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.seed import seed_everything
import torch
from tqdm import tqdm
from datasets import load_from_disk

class QueueSetting(Callback):
    def __init__(self):
        pass 

    def on_fit_start(self, trainer, pl_module):
        seed_everything(42)
        
        wiki_dataset = load_from_disk('/opt/ml/input/data/data/my_wikipedia_1')
        embeddings = torch.tensor(wiki_dataset['embeddings']).to('cuda') * 0.999
        pl_module.queue = embeddings.T
        print('Queue Setting End!')

        # trainer.validate() 
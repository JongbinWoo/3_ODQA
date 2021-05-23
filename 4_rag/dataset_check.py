#%%
import pandas as pd
import json
import re
with open('/opt/ml/input/data/data/wikipedia_documents.json', 'r') as f:
    wiki_data = pd.DataFrame(json.load(f)).transpose()  
# %%

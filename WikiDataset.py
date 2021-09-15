
# coding: utf-8

# ## List of All dataset

# In[6]:


from datasets import load_dataset
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false" ##I will find a way to fix this later :(
#os.environ["TOKENIZERS_PARALLELISM"] = "true" ##I will find a way to fix this later :(


import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoConfig
from transformers import DataCollatorForLanguageModeling


# In[7]:


def getWiki(DIR):
    
    cache_dir=DIR
    
    dataset=load_dataset('wikipedia', '20200501.en', beam_runner='DirectRunner', cache_dir=cache_dir)
    
    return dataset


# In[8]:


def test():
    
    if os.uname()[1].find('gilbreth')==0: ##if not darwin(mac/locallaptop)
        DIR='/scratch/gilbreth/das90/Dataset'
    elif os.uname()[1].find('unimodular')==0:
        DIR='/scratch2/das90/Dataset'
    elif os.uname()[1].find('Siddharthas')==0:
        DIR='/Users/siddharthashankardas/Purdue/Dataset'  
    else:
        DIR='./Results'
        
    cache_dir=DIR
    
    dataset=getWiki(cache_dir)
    
    return dataset


# In[9]:


dataset=test()


# In[10]:


dataset['train'].shape


# In[11]:


dataset['train']


# In[12]:


def encode(examples):
    return berttokenizer(examples['text'], truncation=True)

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, berttokenizer, collator, dataset):
        self.tokenizer = berttokenizer
        self.dataset = dataset
        self.collator = collator

    def __getitem__(self, idx):
        item = encode(dataset['train'][idx])        
        item = self.collator([item])
        item = {key: val[0] for key, val in item.items()}
    
        return item

    def __len__(self):
        return dataset.num_rows
        
A=AutoTokenizer
berttokenizer=A.from_pretrained('bert-base-uncased')
datacollator=DataCollatorForLanguageModeling(tokenizer=berttokenizer,mlm_probability=0.15, mlm=True)

wikidataset = WikiDataset(berttokenizer,datacollator, dataset)

wikidataset[4]


# In[13]:


dataset.num_rows['train']


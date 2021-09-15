
# coding: utf-8

# ### Packages

# In[1]:


import torch# If there's a GPU available...
import random
import numpy as np
import multiprocessing
import pandas as pd
import time

import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false" ##I will find a way to fix this later :(

NUM_GPUS=0

try:
    if torch.cuda.is_available():  
        device = torch.device("cuda")
        NUM_GPUS=torch.cuda.device_count()
        print('There are %d GPU(s) available.' % NUM_GPUS)
        print('We will use the GPU:', torch.cuda.get_device_name())# If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")  
except:
    print('Cuda error using CPU instead.')
    device = torch.device("cpu")  
    
print(device)

# device = torch.device("cpu")  
# print(device)

NUM_PROCESSORS=multiprocessing.cpu_count()
print("Cpu count: ",NUM_PROCESSORS)


# #### Specify Directories

# In[2]:


#from ipynb.fs.full.Dataset import getDataset, getDummyDataset, Data        

if os.uname()[1].find('gilbreth')==0: ##if not darwin(mac/locallaptop)
    DIR='/scratch/gilbreth/das90/Dataset'
elif os.uname()[1].find('unimodular')==0:
    DIR='/scratch2/das90/Dataset'
elif os.uname()[1].find('Siddharthas')==0:
    DIR='/Users/siddharthashankardas/Purdue/Dataset'  
else:
    DIR='./Results'
    
from pathlib import Path
Path(DIR).mkdir(parents=True, exist_ok=True)

MODEL_SAVE_DIR=DIR+'/NVD/Model/'

Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
print("Data loading directory: ", DIR)
print("Model Saving directory:", MODEL_SAVE_DIR)


# In[3]:


import pandas as pd
import pathlib
import zipfile
import wget

import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint


# #### For reproduciblity

# In[4]:


# Set the seed value all over the place to make this reproducible.
from random import sample

seed_val = 42
os.environ['PYTHONHASHSEED'] = str(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
pl.seed_everything(seed_val)

try:
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    print("nothing to set for cudnn")


# ### Model definition

# In[5]:


class Model(pl.LightningModule):
    def __init__(self,*args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        # a very useful feature of pytorch lightning  which leads to the named variables that are passed in
        # being available as self.hparams.<variable_name> We use this when refering to eg
        # self.hparams.learning_rate

        # freeze
        self._frozen = False

        # eg https://github.com/stefan-it/turkish-bert/issues/5
        config = AutoConfig.from_pretrained(self.hparams.pretrained,
                                            output_attentions=False,
                                            output_hidden_states=False)

        #print(config)

        A = AutoModelForMaskedLM
        self.model = A.from_pretrained(self.hparams.pretrained, config=config)

        print('Model: ', type(self.model))
        

    def forward(self, batch):
        # there are some choices, as to how you can define the input to the forward function I prefer it this
        # way, where the batch contains the input_ids, the input_put_mask and sometimes the labels (for
        # training)
        
        #print(batch['input_ids'].shape)
        #print(batch['labels'].shape)
                
        outputs = self.model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])

        loss = outputs[0]

        return loss
        
    def training_step(self, batch, batch_idx):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition
        
        # self refers to the model, which in turn acceses the forward method
        loss = self(batch)
        
        #tensorboard_logs = {'train_loss': loss}
        # pytorch lightning allows you to use various logging facilities, eg tensorboard with tensorboard we
        # can track and easily visualise the progress of training. In this case
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        #return {'loss': loss, 'log': tensorboard_logs}
        # the training_step method expects a dictionary, which should at least contain the loss
        return loss
    
#     def validation_step(self, batch, batch_idx):
#         val_loss = self(batch)
#         self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

#         return val_loss
        
#     def test_step(self, batch, batch_idx):
#         test_loss = self(batch)
#         self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        
#         return test_loss
    
    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.
        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=1e-8 # args.adam_epsilon  - default is 1e-8.
                          )

        
        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0, # Default value in run_glue.py
            num_training_steps=self.hparams.num_training_steps)

        return [optimizer], [scheduler]


# ### Data class definition

# In[6]:


from datasets import load_dataset

def getWiki(DIR):
    
    cache_dir=DIR
    
    dataset=load_dataset('wikipedia', '20200501.en', beam_runner='DirectRunner', cache_dir=cache_dir)
    
    return dataset


class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, berttokenizer, collator, dataset):
        self.tokenizer = berttokenizer
        self.dataset = dataset
        self.collator = collator

    def __getitem__(self, idx):
        item = self.tokenizer(self.dataset['train'][idx]['text'], truncation=True, padding='max_length')  
        item = self.collator([item])
        item = {key: val[0] for key, val in item.items()}
    
        return item

    def __len__(self):
        return self.dataset.num_rows['train']
        #return 2000
        
# A=AutoTokenizer
# berttokenizer=A.from_pretrained('bert-base-uncased')
# datacollator=DataCollatorForLanguageModeling(tokenizer=berttokenizer,mlm_probability=0.15, mlm=True)
# wikidataset = WikiDataset(berttokenizer,datacollator, dataset)
# wikidataset[4]


# In[7]:


class DataProcessing(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

#         self.save_hyperparameters()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        self.batch_size=self.hparams.batch_size        

        #print('Loading BERT tokenizer')
        print(f'PRETRAINED:{self.hparams.pretrained}')

        A = AutoTokenizer
        self.tokenizer = A.from_pretrained(self.hparams.pretrained, use_fast=True)

        print('Tokenizer:', type(self.tokenizer))
        
        self.datacollator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        
    def setup(self, stage=None):
        dataset=getWiki(DIR)
        self.wikidataset = WikiDataset(self.tokenizer,self.datacollator, dataset)        
    
    def train_dataloader(self):
        
        train_sampler = RandomSampler(self.wikidataset)
        
        return DataLoader(self.wikidataset,
                         #sampler=train_sampler, 
                          batch_size=self.batch_size,
                          num_workers=min(NUM_PROCESSORS,self.batch_size,4)
                         )


# In[8]:


def printModelParams(model):
    print (model)
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))

    print('\n==== Output Layer ====\n')
    for p in params[-5:]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))


# In[9]:


def print_model_value(model):
    params = list(model.named_parameters())
    print (params[-1][0],params[-1][1][:4])


# ### Get Configuration to run

# In[10]:


import argparse
from argparse import ArgumentParser

def get_configuration():
    parser = ArgumentParser()

    parser.add_argument('--pretrained', type=str, default="bert-base-uncased")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--nr_frozen_epochs', type=int, default=5)
    parser.add_argument('--training_portion', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--auto_batch', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--parallel_mode', type=str, default="dp", choices=['dp', 'ddp', 'ddp2'])
    parser.add_argument('--refresh_rate', type=int, default=1)
    parser.add_argument('--check', type=bool, default=False)
    parser.add_argument('--rand_dataset', type=str, default="dummy", choices=['temporal','random','dummy'])
    
    
    parser.add_argument('-f') ##dummy for jupyternotebook

    # parser = Model.add_model_specific_args(parser) parser = Data.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    print("-"*50)
    print("BATCH SIZE: ", args.batch_size)
    
    # start : get training steps
    dataProcessor = DataProcessing(args)
    dataProcessor.setup()
    
    args.num_training_steps = len(dataProcessor.train_dataloader())*args.epochs
    dict_args = vars(args)
    
    gpus=-1
    if NUM_GPUS>0:
        gpus=args.num_gpus        
    else:
        args.parallel_mode=None
        gpus=None
    
    print("USING GPUS:", gpus)
    print("-"*50)
    
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='loss_epoch',
        dirpath=MODEL_SAVE_DIR,
        filename="V2W-"+args.pretrained+'-'+str(args.parallel_mode),
        save_top_k=1,
        mode='min',
        save_weights_only=True,
        #prefix="V2W-"+args.pretrained+'-'+str(args.parallel_mode),
        save_last=True,
    )
    
    if args.check==False:
        args.checkpoint_callback = False
    elif args.parallel_mode=='dp':
        args.callbacks=[checkpoint_callback]        
    else:
        args.checkpoint_callback = False
    
    trainer = pl.Trainer.from_argparse_args(args, 
                                            gpus=gpus,
                                            num_nodes=args.nodes, 
                                            accelerator=args.parallel_mode,
                                            max_epochs=args.epochs, 
                                            gradient_clip_val=1.0,                                            
                                            logger=False,
                                            progress_bar_refresh_rate=args.refresh_rate,
                                            profiler='simple', #'simple',
                                            default_root_dir=MODEL_SAVE_DIR,                                            
                                            deterministic=True,
                                           )

    return trainer, dataProcessor, args, dict_args

# trainer, dataProcessor, args, dict_args = get_configuration()
# next(iter(dataProcessor.test_dataloader()))


# In[11]:


def train_model():    
    trainer, dataProcessor, args, dict_args = get_configuration()
    
    model = Model(**dict_args)    
    
#     printModelParams(model)
#     args.early_stop_callback = EarlyStopping('val_loss')

    print("Original weights: ");print_model_value(model)
    
    t0=time.time()
    trainer.fit(model, dataProcessor)
    print('Training took: ',time.time()-t0)
    
    print("Trained weights: ");print_model_value(model)
    
#     if args.parallel_mode!='dp':    
#         print("Saving the last model")
#         #MODEL_NAME=MODEL_SAVE_DIR+"V2W-"+args.pretrained+'-'+args.parallel_mode+".ckpt"
#         MODEL_NAME=MODEL_SAVE_DIR+"V2W-"+args.pretrained+".ckpt"
#         trainer.save_checkpoint(MODEL_NAME)

#     print("Testing:....")
#     trainer.test(model, dataProcessor.test_dataloader())
    
    print("Training Phase Complete......")


# In[12]:


def test_model():
    trainer, dataProcessor, args, dict_args = get_configuration()
    
    #MODEL_NAME=MODEL_SAVE_DIR+"V2W-"+args.pretrained+'-'+args.parallel_mode+".ckpt"    
    MODEL_NAME=MODEL_SAVE_DIR+"V2W-"+args.pretrained+".ckpt"    
    if args.parallel_mode=='dp':
        MODEL_NAME=MODEL_SAVE_DIR+"V2W-"+args.pretrained+'-'+args.parallel_mode+"-last.ckpt"
    
    if os.path.exists(MODEL_NAME): 
        print('Loading Saved Model: ',MODEL_NAME)        
    else: 
        print("File not found: ",MODEL_NAME)
        return
    
    model=None
    
    if args.parallel_mode!='dp':
        model = Model.load_from_checkpoint(MODEL_NAME)
    else:
        model = Model(**dict_args)
        print("Original weights: ");print_model_value(model)
        checkpoint = torch.load(MODEL_NAME, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    print("Loaded weights: ");print_model_value(model)    
    trainer.test(model, dataProcessor.test_dataloader())    
    print("Test Complete......")
    
    return model


# In[13]:


if __name__ == "__main__":
    train_model()
    #test_model()    


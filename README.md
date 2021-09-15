<!-- https://gist.github.com/PurpleBooth/109311bb0361f32d87a2 -->
<!-- https://pandao.github.io/editor.md/en.html -->

# BERT Pretraining


### Necessary Libraries:

Pytorch latest version 

```
pip install transformers
pip install pytorch-lightning
pip install datasets
pip install ipynb
pip install pandas
```

There might be some are other libraries necesasry for installation


### Settings:

The WikiPedia dataset takes about 18GB space. Therefore specify appropriate directory for the dataset.

1. In the `WikiDataset.ipynb` or `WikiDataset.py` change the DIR in line `44-51`
2. In the `8-V2W-Pretraining-Wiki.ipynb` or `8-V2W-Pretraining-Wiki.py` change the DIR in lines `50-57`

##### Notes:
In `8-V2W-Pretraining-Wiki` file you can set `#return 2000` to work with only 2000 data. Since this dataset contains more than 6million entry, this could helpful as a startup code to test.

### Command to execute:

- The extension `.py` is for executable python file and `.ipynb` is the corresponding jupyter notebook.



Run this one setting appropriate directory in the code to download dataset

```
python WikiDataset.py
```


Run this one 1 epoch to get the runtime. 


```
python 8-V2W-Pretraining-Wiki.py --pretrained='bert-large-uncased' --epochs=1 --nodes=1 --num_gpus=2 --batch_size=8 --parallel_mode='ddp' --refresh_rate=1000
```


##### Commands interpretation:

`--batch_size=16` %%%most important one, change according to the GPU capacity
`--parallel_mode='ddp'` %%% in distributed data parallel, batch size  8 means, each GPU will handle 8 data, choices are "dp" and "ddp"

`--pretrained='bert-large-uncased'` for BERT Large model, e.g. 'bert-base-uncased' for base
`--epochs=1` run one epoch
`--nodes=1` run in one node
`--num_gpus=2` run in how many GPUs
`--refresh_rate=1000` after 1000 minibatch processing the result will be at console, this is to stop too many outouts, you can set it to 0 to stop printing those
`--check=True` default is false, set it true to save model if necessary



## Authors

**Siddhartha Shankar Das**

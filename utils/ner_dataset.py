import torch
import os
import json
class NER_dataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, input_file = None, output_file = None, split='fine-tune', label_type ='raw'):
    if input_file is None:
      input_file = f'data/slue_dataset/{split}.wrd'
    if output_file is None:
      output_file = f'data/slue_dataset/{split}.{label_type}.wrd'
    with open("data/ner_dict.json") as f:
      ner_dict = json.load(f)
    with open(input_file) as f:
      inputs= f.read().splitlines()
    with open(output_file) as f:
      targets = f.read().splitlines()
    for i in range(len(targets)):
      if targets[i][0] in ner_dict.values():
        targets[i] = ' ' + targets[i]
      
    self.inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    self.targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt").input_ids
    self.targets[self.targets == tokenizer.pad_token_id] = -100
    #self.target_lens = (self.targets==1).nonzero(as_tuple = True)[1]
  
  def __getitem__(self, idx):
    item = {"input_ids": self.inputs.input_ids[idx],
            "attention_mask": self.inputs.attention_mask[idx],
            "labels" : self.targets[idx] }
    return item

  def __len__(self):
    return len(self.targets)
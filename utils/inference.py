import torch
def infer(model, tokenizer, dataset, save_as, batch_size=64,):
  device = model.device
  with open(save_as, 'w') as f:
    model.eval()
    with torch.no_grad():
      loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
      count = 0
      for batch in loader:
        count += 1
        #print(count)
        outputs = model.generate(input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        do_sample=False,
        max_length = 200)
        sens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        f.writelines(map(lambda x: x+'\n', sens))
    model.train()
  #return sens
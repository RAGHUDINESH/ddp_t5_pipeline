import os
from tqdm.auto import tqdm
import numpy as np
import torch

def trainer(model, train_dataset, eval_dataset, ckpt_dir, save_id,
        epochs = 100,
        eval_every = 3,
        lr = 1e-5,
        batch_size = 16,
        eval_batch_size = 16):

  device = model.device  
  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size)

  optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

  try:
    os.mkdir(ckpt_dir)
  except:
    print('Checkpoint Directory Already Exists')
  
  min_eval_loss = 1000000
  for epoch in tqdm(range(epochs), ascii=True, desc="Training Epochs", position=0):
    model.train()
    train_losses = []
    for batch in tqdm(trainloader,  ascii=True, desc=f"Training: Epoch {epoch+1}", position=1, leave=False):
      optimizer.zero_grad()
      loss = model(input_ids = batch['input_ids'].to(device), attention_mask = batch["attention_mask"].to(device), labels = batch["labels"].to(device)).loss
      loss.backward()
      optimizer.step()
      train_losses += [loss.item()]
    print(f'epoch {epoch+1}: \t training loss: {np.average(train_losses)}')
    
    if epoch%eval_every == 0:
      model.eval()
      eval_losses = []
      with torch.no_grad():
        for batch in tqdm(evalloader,  ascii=True, desc=f"Evaluation after epoch {epoch}", position=1, leave=False):
          eval_loss = model(input_ids = batch['input_ids'].to(device), attention_mask = batch["attention_mask"].to(device) , labels = batch["labels"].to(device)).loss.item()
          eval_losses += [eval_loss]
        eval_net_loss = np.average(eval_losses)
      print(f"\t \t Evaluation Loss: {eval_net_loss}")
      
      if eval_net_loss < min_eval_loss:
        min_eval_loss = eval_net_loss
        torch.save(model.state_dict(), os.path.join(ckpt_dir,f'best_checkpoint_inline_T5_{save_id}.pt'))
        print(f'Epoch {epoch+1} saved as best checkpoint')
      
  model.train()
  torch.save(model.state_dict(), os.path.join(ckpt_dir, f'final_checkpoint_inline_T5_{save_id}.pt'))

import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as Scheduler

def train(model, 
          batch_size, 
          train_loader,
          valid_loader=None,
          warm_up_epoch=20, 
          num_epochs=30, 
          weight_decay=1e-6,
          learning_rate=1e-2,
          seed=1, 
          verbose=True,
          patience=10,
          save_path = "train_with_tsne/"):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )   

  scheduler = Scheduler.CosineAnnealingLR(optimizer, num_epochs)
  patience_count = 0
  best_loss = 10000

  for epoch in range(num_epochs):
    if verbose:
      print("Learning Rate = {0}".format(optimizer.param_groups[0]['lr']))
    model.warm_up_weight = min(epoch / warm_up_epoch, 1.0)

    for i, sample in enumerate(train_loader):
      x = sample['x'].to(device=device)
      
      optimizer.zero_grad()
      loss, kl_divergence_z, kl_divergence_y, reconstruction_error = model(x)
      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), 5)
      optimizer.step()

      if i % 5 == 0 and verbose:
          print("Training: Epoch[{}/{}], Step [{}/{}],  Loss: {:.4f}, KL Div Z: {:.4f}, KL Div Y: {:.4f}, Recon Loss: {:.4f}".format(
                                  epoch + 1, num_epochs, i, len(train_loader), loss.item(), kl_divergence_z.item(), kl_divergence_y.item(), reconstruction_error.item()))
    if valid_loader:
      tot_loss = 0
      for i, sample in enumerate(valid_loader):
        with torch.no_grad():
          model.eval()
          x = sample['x'].to(device=device)
          loss, kl_divergence_z, kl_divergence_y, reconstruction_error = model(x)
          model.train()
          tot_loss += loss.item()

          if i % 5 == 0 and verbose:
            print("Validation: Epoch[{}/{}], Step [{}/{}],  Loss: {:.4f}, KL Div Z: {:.4f}, KL Div Y: {:.4f}, Recon Loss: {:.4f}".format(
                                  epoch + 1, num_epochs, i, len(valid_loader), loss.item(), kl_divergence_z.item(), kl_divergence_y.item(), reconstruction_error.item()))

      avg_loss = tot_loss/len(valid_loader)
      print("Average Validation Loss = {}".format(avg_loss))
      if (avg_loss < best_loss) :
        best_loss = avg_loss
        patience_count = 0
      else:
        patience_count += 1

      if patience_count == patience:
        print("Early Stopping")
        return best_loss

    scheduler.step()

  return best_loss
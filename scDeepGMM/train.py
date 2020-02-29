import torch
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
from .utils import NMI
import numpy as np
def train(model, train_loader, warm_up_epoch = 20, num_epochs = 30, weight_decay = 1e-6, learning_rate = 1e-2,
    seed = 1, verbose = True, patience = 10, file_ind = False, NMI_ind = True, warm_up_clusters = 5.,
    mmd_ind = True,mmd_weight = 1.,mmd_epoch = float("Inf"),load_file = None, save_file = None):
    
    file_dir = os.getcwd()

    cudnn.benchmark = True
    cudnn.enabled = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    patience_count = 0
    best_loss = float("Inf")
    start = 0 

    if load_file:
        checkpoint = torch.load(os.path.join(file_dir,'modelDeepGMM', load_file)) 
        print("Loading Checkpoint")
        model.load_state_dict(checkpoint["State Dict"])
        scheduler.load_state_dict(checkpoint["Scheduler"])
        # model.load_state_dict(checkpoint["State Dict"])
        optimizer.load_state_dict(checkpoint["Optimizer"])
        patience = checkpoint["Patience"]
        best_loss = checkpoint["Best Loss"]
        start = checkpoint["Epoch"]

    # for epoch in range(start,5):
    #     for i, sample in enumerate(train_loader):
    #         x = sample['x']
    #         optimizer.zero_grad()
    #         # if i == 1:
    #         #     model.init_gmm(x)
    #         #     pass
    #         if mmd_ind is True:
    #             batch_ids = sample['batch']
    #             likelihood, kld_z, kld_c, mmd = model(x, batch_ids)
    #         else:
    #             likelihood, kld_z, kld_c, mmd = model(x)

    #         loss = likelihood
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), 5.)
    #         optimizer.step()

    #         latent_y = model.get_latent_y(x)

    #         if NMI_ind:
    #             score = NMI(torch.argmax(latent_y,1).cpu().detach().numpy(),
    #                 sample['labels'])
    #             print("Score: {:.4f}".format(score))

    #         if verbose:
    #             print("Training: Epoch[{}/{}], Step [{}/{}],  Loss: {:.4f}, KL Z: {:.4f}, KL Y: {:.4f}, Recon Loss: {:.4f}, MMD: {:.4f}, Mean prob: {:.4f}".format(
    #                                                                 epoch + 1, num_epochs, i, len(train_loader), loss.item(), 
    #                                                                 kld_z.item(), kld_c.item(), likelihood.item(),mmd.item(), latent_y.max(1).values.mean().item()))
    #         if np.isnan(latent_y.max(1).values.mean().item()) is True:
    #             return latent_y

    # x = iter(train_loader).next()['x']
    # model.init_gmm(x)
    max_prob = 0.
    for epoch in range(start, num_epochs):
        
        print("Learning Rate = {0}".format(optimizer.param_groups[0]['lr']))
        
        warm_up_weight_z = min(epoch/warm_up_epoch, 1.0)
        warm_up_weight_c = min(epoch/(warm_up_epoch * warm_up_clusters), 1.)

        running_score = 0

        for i, sample in enumerate(train_loader):
            x = sample['x']
            optimizer.zero_grad()
            # if i == 1:
            #     model.init_gmm(x)
            #     pass
            if mmd_ind is True:
                batch_ids = sample['batch']
                likelihood, kld_z, kld_c, mmd = model(x, batch_ids)
                latent_y = model.get_latent_y(x, batch_ids)
                if epoch < mmd_epoch:
                	loss = likelihood + warm_up_weight_z * kld_z + warm_up_weight_c * kld_c 
                else:
                	loss = likelihood + warm_up_weight_z * kld_z + warm_up_weight_c * kld_c + mmd_weight * mmd
            else:
                likelihood, kld_z, kld_c, mmd = model(x)
                latent_y = model.get_latent_y(x)
                loss = likelihood + warm_up_weight_z * kld_z + warm_up_weight_c * kld_c

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()


            if NMI_ind:
                score = NMI(torch.argmax(latent_y,1).cpu().detach().numpy(),
                    sample['labels'])
                print("Score: {:.4f}".format(score))
            else:
            	print(f'Clusters: {torch.argmax(latent_y,1).cpu().detach().unique()}')
            curr_prob = latent_y.max(1).values.mean().item()

            if epoch % 1 == 0:
                print("Training: Epoch[{}/{}], Step [{}/{}],  Loss: {:.4f}, KL Z: {:.4f}, KL Y: {:.4f}, Recon Loss: {:.4f}, MMD: {:.4f}, Mean prob: {:.4f}".format(
                                                                        epoch + 1, num_epochs, i, len(train_loader), loss.item(), 
                                                                        kld_z.item(), kld_c.item(), likelihood.item(),mmd.item(), curr_prob))
            if np.isnan(latent_y.detach().numpy()).any() is True:
                return latent_y

            # if max_prob < curr_prob:
            #     curr_prob = max_prob
            #     patience_count = 0
            # else:
            #     patience_count += 1
            #     print(patience_count)
            # if patience_count == patience:
            #     print("Early Stopping")
            #     return max_prob

        if save_file:
            checkpoint = {
                "Model"      : model,
                "State Dict" : model.state_dict(),
                "Optimizer"  : optimizer.state_dict(),
                "Scheduler"  : scheduler.state_dict(),
                "Patience"   : patience,
                "Best Loss"  : best_loss,
                "Epoch"      : epoch,
            }
            torch.save(checkpoint, os.path.join(file_dir,'modelDeepGMM', save_file))

        scheduler.step()
    return loss.item()


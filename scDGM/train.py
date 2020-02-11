import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as Scheduler
import numpy as np
import torch.backends.cudnn as cudnn
import datetime

from .model import GMVAE
from .evaluation import ARI, NMI

def train(model,
                train_loader,
                valid_loader=None,
                warm_up_epoch=20, 
                num_epochs=30, 
                weight_decay=1e-6,
                learning_rate=1e-2,
                seed=1, 
                verbose=True,
                patience=10,
                file_ind = False,
                NMI_ind = True,
                save_file = 'Model_Checkpoint.pt'
            ):

    file_dir = os.path.join(os.getcwd())

    cudnn.benchmark = True
    cudnn.enabled=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(
                        model.parameters(), 
                        lr=learning_rate, 
                        weight_decay=weight_decay
                )   

    scheduler = Scheduler.CosineAnnealingLR(optimizer, num_epochs)
    patience_count = 0
    best_loss = float("Inf")
    start = 0
    
    if file_ind:
        for file in os.listdir(os.path.join(file_dir, 'checkpoints')):
            if save_file in file:
                checkpoint = torch.load(os.path.join(file_dir, 'checkpoints', file)) 
                print("Loading Checkpoint")
                model.load_state_dict(checkpoint["State Dict"])
                scheduler.load_state_dict(checkpoint["Scheduler"])
                model.load_state_dict(checkpoint["State Dict"])
                optimizer.load_state_dict(checkpoint["Optimizer"])
                patience = checkpoint["Patience"]
                best_loss = checkpoint["Best Loss"]
                start = checkpoint["Epoch"]

    for epoch in range(start, num_epochs):
        if verbose:
            print("Learning Rate = {0}".format(optimizer.param_groups[0]['lr']))
        model.warm_up_weight = min(epoch / warm_up_epoch, 1.0)
        running_score = 0
        for i, sample in enumerate(train_loader):
            x = sample['x'].to(device=device)
            
            optimizer.zero_grad()
            loss, kl_divergence_z, kl_divergence_y, reconstruction_error = model(x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            latent_y = model.get_latent_y(x)  # Latent y

            
            if NMI_ind:
                guesses = torch.argmax(latent_y, dim=1)
                score = NMI(guesses.cpu().detach().numpy(), sample['labels'])
                running_score += score
            if verbose:
                checkpoint = {
                    "Model"      : model,
                    "State Dict" : model.state_dict(),
                    "Optimizer"  : optimizer.state_dict(),
                    "Scheduler"  : scheduler.state_dict(),
                    "Patience"   : patience,
                    "Best Loss"  : best_loss,
                    "Epoch"      : epoch,
                }
                print("Mean prob: ",latent_y.max(1).values.mean().item())
                torch.save(checkpoint, os.path.join(file_dir,'checkpoints', save_file))
                print("Training: Epoch[{}/{}], Step [{}/{}],  Loss: {:.4f}, KL Div Z: {:.4f}, KL Div Y: {:.4f}, Recon Loss: {:.4f}, Score: {}".format(
                                                                    epoch + 1, num_epochs, i, len(train_loader), loss.item(), kl_divergence_z.item(), kl_divergence_y.item(), reconstruction_error.item(), running_score /(i+1)))
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
            print("Epoch[{}/{}]: Average Validation Loss = {}".format(epoch + 1, num_epochs, avg_loss))
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

def randomSearch(
                                dataMgr,
                                test_params, 
                                batch_size, 
                                warm_up_epoch=20, 
                                num_epochs=30, 
                                weight_decay=1e-6,
                                learning_rate=1e-2,
                                seed=1, 
                                verbose=False,
                                patience=10,
                                iterations=10, 
                                scoring='NMI'
                            ):
    best = 0

    for search in range(iterations):
        if 'n_hidden' in test_params.keys():
            n_hidden = np.random.randint(low=test_params['n_hidden'][0], high=test_params['n_hidden'][1])
        else:
            n_hidden = 200

        if 'latent_size' in test_params.keys():
            latent_size = np.random.randint(low=test_params['latent_size'][0], high=test_params['latent_size'][1])
        else:
            latent_size = 200

        if 'kl_weight' in test_params.keys():
            kl_weight = np.random.uniform(low=test_params['kl_weight'][0], high=test_params['kl_weight'][1])
        else:
            kl_weight = 1

        if 'epochs' in test_params.keys():
            epochs = np.random.randint(low=test_params['epochs'][0], high=test_params['epochs'][1])
        else:
            epochs = 30


        print("Test Num {0}: n_hidden={1}, latent_size={2}, kl_weight={3}, epochs={4}".format(search, n_hidden, latent_size, kl_weight, epochs) )

        gmvae = GMVAE(
            n_input=dataMgr.getX().size(1), 
            n_hidden=n_hidden, 
            latent_size=latent_size, 
            n_clusters=7,
            kl_weight=kl_weight
        )

        if 'valid_loader' in dir(dataMgr):
            train(
                gmvae, 
                batch_size=batch_size,
                train_loader=dataMgr.train_loader,
                valid_loader=dataMgr.valid_loader,
                warm_up_epoch=warm_up_epoch,  
                num_epochs=epochs,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                seed=seed, 
                verbose=verbose,
                patience=patience,
            )
        else: 
            train(
                gmvae, 
                batch_size=batch_size,
                train_loader=dataMgr.train_loader,
                warm_up_epoch=warm_up_epoch,  
                num_epochs=epochs,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                seed=seed, 
                verbose=verbose,
                patience=patience,
            )

        latent_y = gmvae.get_latent_y(torch.Tensor(dataMgr.getX()).cuda())
        guesses = torch.argmax(latent_y.probs, dim=1)
        if scoring == 'NMI':
            score = NMI(guesses.cpu().detach().numpy(), dataMgr.getRawData().obs['precise_clusters'])
        elif scoring == 'ARI':
            score = ARI(guesses.cpu().detach().numpy(), dataMgr.getRawData().obs['precise_clusters'])

        if score > best:
            best = score
            opt_params = {
                    'n_hidden'    : n_hidden,
                    'latent_size' : latent_size,
                    'kl_weight'   : kl_weight,
                    'epochs'      : epochs,
                }
        print("Score: {0}, Best Score: {1}".format(score, best))
    return best, opt_params

def gridSearch(
                                dataMgr,
                                test_params, 
                                batch_size, 
                                warm_up_epoch=20, 
                                num_epochs=30, 
                                weight_decay=1e-6,
                                learning_rate=1e-2,
                                seed=1, 
                                verbose=False,
                                patience=10,
                                scoring='NMI'
                            ):
    best = 0

    if 'n_hidden' in test_params.keys():
        n_hidden_ = np.arange(
                start=test_params['n_hidden'][0], 
                stop=test_params['n_hidden'][1]
            )
    else:
        n_hidden_ = np.array([200])

    if 'latent_size' in test_params.keys():
        latent_size_ = np.arange(
                start = test_params['latent_size'][0], 
                stop  = test_params['latent_size'][1]
            )
    else:
        latent_size_ = np.array([200])

    if 'kl_weight' in test_params.keys():
        kl_weight_ = np.arange(
                start = test_params['kl_weight'][0], 
                stop  = test_params['kl_weight'][1],
                step  = test_params['kl_weight'][2]
            )
    else:
        kl_weight_ = np.array([1])

    if 'epochs' in test_params.keys():
        epochs_ = np.arange(
                start = test_params['epochs'][0], 
                stop  = test_params['epochs'][1]
            )
    else:
        epochs_ = np.array([30])

    num_params = len(epochs_) * len(kl_weight_) * len(latent_size_) * len(n_hidden_)
    counter = 0

    for n_hidden in n_hidden_:
        for latent_size in latent_size_:
            for kl_weight in kl_weight_:
                for epochs in epochs_:
                    counter += 1
                    print("Test Num {}/{}: n_hidden={}, latent_size={}, kl_weight={}, epochs={}".format(counter, num_params, n_hidden, latent_size, kl_weight, epochs) )

                    gmvae = GMVAE(
                        n_input=dataMgr.getX().size(1), 
                        n_hidden=n_hidden, 
                        latent_size=latent_size, 
                        n_clusters=7,
                        kl_weight=kl_weight
                    )

                    if 'valid_loader' in dir(dataMgr):
                        train(
                            gmvae, 
                            batch_size=batch_size,
                            train_loader=dataMgr.train_loader,
                            valid_loader=dataMgr.valid_loader,
                            warm_up_epoch=warm_up_epoch,  
                            num_epochs=epochs,
                            weight_decay=weight_decay,
                            learning_rate=learning_rate,
                            seed=seed, 
                            verbose=verbose,
                            patience=patience,
                        )
                    else: 
                        train(
                            gmvae, 
                            batch_size=batch_size,
                            train_loader=dataMgr.train_loader,
                            warm_up_epoch=warm_up_epoch,  
                            num_epochs=epochs,
                            weight_decay=weight_decay,
                            learning_rate=learning_rate,
                            seed=seed, 
                            verbose=verbose,
                            patience=patience,
                        )

                    latent_y = gmvae.get_latent_y(torch.Tensor(dataMgr.getX()).cuda())
                    guesses = torch.argmax(latent_y.probs, dim=1)
                    if scoring == 'NMI':
                        score = NMI(guesses.cpu().detach().numpy(), dataMgr.getRawData().obs['precise_clusters'])
                    elif scoring == 'ARI':
                        score = ARI(guesses.cpu().detach().numpy(), dataMgr.getRawData().obs['precise_clusters'])

                    if score > best:
                        best = score
                        opt_params = {
                                'n_hidden'    : n_hidden,
                                'latent_size' : latent_size,
                                'kl_weight'   : kl_weight,
                                'epochs'      : epochs,
                            }
                    print("Score: {0}, Best Score: {1}".format(score, best))
    return best, opt_params
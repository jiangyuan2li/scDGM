import torch.nn as nn
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.1)
        else:
            nn.init.constant_(param.data, 0)
import torch
import numpy as np

def detection_collate(batch):
    imgs = []
    targets = []
    for img, target in batch:
        imgs.append(torch.from_numpy(img))
        targets.append(torch.from_numpy(target))

    return torch.stack(imgs, 0), targets

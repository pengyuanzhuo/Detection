import torch
from data import Resize, VOCDetection, detection_collate

if __name__ == '__main__':
    transform = Resize(300)
    vocdataset = VOCDetection('/workspace/dataset/VOCdevkit', transform=transform)
    vocloader = torch.utils.data.DataLoader(vocdataset, batch_size=4, shuffle=True,
                                            collate_fn=detection_collate)
    for img, target in vocloader:
        print(img.shape)
        print(target)
        break

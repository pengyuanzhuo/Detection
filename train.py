import torch
import data.voc as voc
import data.collate as collate


if __name__ == '__main__':
    transform = aug.Resize(300)
    vocdataset = VOCDetection('/workspace/dataset/VOCdevkit', transform=transform)
    vocloader = data.DataLoader(vocdataset, batch_size=4, shuffle=True, collate_fn=collate.detection_collate)
    for img, target in vocloader:
        print(img.shape)
        print(target)
        break
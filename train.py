import torch
from data import Resize, VOCDetection, detection_collate
from models.ssd import build_ssd
from config import Config as cfg

if __name__ == '__main__':
    model = build_ssd(cfg)
    inputs = torch.zeros((2, 3, 300, 300))
    print(inputs.shape)
    default_box, conf, loc = model(inputs)
    print(default_box.shape)

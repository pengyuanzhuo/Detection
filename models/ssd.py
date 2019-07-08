# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.proposal.default_box import DefaultBox


base_cfg = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

extras_cfg = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}


def make_vgg(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for c in cfg:
        if c == 'M':
            layers += [nn.MaxPool2d(2, stride=2)]
        elif c == 'C':
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(input_channel, c, kernel_size=3, stride=1, padding=1)
            input_channel = c
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

    pool5 = nn.MaxPool2d(3, stride=1, padding=1) # 1024*19*19
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # 1024*19*19
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1) # 1024*19*19
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def make_extras():
    # conv8
    extra1_1 = nn.Conv2d(1024, 256, kernel_size=1) # (256, 19, 19)
    extra1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # (512, 10, 10)

    # conv9
    extra2_1 = nn.Conv2d(512, 128, kernel_size=1) # (128, 10, 10)
    extra2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # (256, 5, 5)

    # conv10
    extra3_1 = nn.Conv2d(256, 128, kernel_size=1) # (128, 5, 5)
    extra3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # (256, 3, 3)

    # conv11
    extra4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # (128, 3, 3)
    extra4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # (256, 1, 1)

    return [extra1_1, extra1_2, extra2_1, extra2_2, extra3_1, extra3_2, extra4_1, extra4_2]


def make_head(vgg, extras, num_classes=21):
    # output feature map [conv4_3, conv7, extra1_2, extra2_2, extra3_2, extra4_2]
    # vgg[21]是conv4_3, vgg[-2]是conv7(relu之前)
    conf4_3 = nn.Conv2d(vgg[21].out_channels, 4*num_classes, kernel_size=3, stride=1, padding=1)
    loc4_3 = nn.Conv2d(vgg[21].out_channels, 4*4, kernel_size=3, stride=1, padding=1)

    conf_7 = nn.Conv2d(vgg[-2].out_channels, 6*num_classes, kernel_size=3, stride=1, padding=1)
    loc_7 = nn.Conv2d(vgg[-2].out_channels, 6*4, kernel_size=3, stride=1, padding=1)

    conf8_2 = nn.Conv2d(extras[1].out_channels, 6*num_classes, 3, 1, 1)
    loc8_2 = nn.Conv2d(extras[1].out_channels, 6*4, 3, 1, 1)

    conf9_2 = nn.Conv2d(extras[3].out_channels, 6*num_classes, 3, 1, 1)
    loc9_2 = nn.Conv2d(extras[3].out_channels, 6*4, 3, 1, 1)

    conf10_2 = nn.Conv2d(extras[5].out_channels, 4*num_classes, 3, 1, 1)
    loc10_2 = nn.Conv2d(extras[5].out_channels, 4*4, 3, 1, 1)

    conf11_2 = nn.Conv2d(extras[7].out_channels, 4*num_classes, 3, 1, 1)
    loc11_2 = nn.Conv2d(extras[7].out_channels, 4*4, 3, 1, 1)

    loc_layers = [loc4_3, loc_7, loc8_2, loc9_2, loc10_2, loc11_2]
    conf_layers = [conf4_3, conf_7, conf8_2, conf9_2, conf10_2, conf11_2]

    return (conf_layers, loc_layers)


class SSD(nn.Module):
    def __init__(self, base_net, extra_net, head, cfg):
        super(SSD, self).__init__()
        self.base = nn.ModuleList(base_net)
        self.extras = nn.ModuleList(extra_net)
        self.conf = nn.ModuleList(head[0])
        self.loc = nn.ModuleList(head[1])
        self.num_classes = cfg.num_classes
        self.default_box_fun = DefaultBox(cfg.input_size, cfg.feature_maps, 
                                          cfg.ratios, cfg.min_scale, cfg.max_scale)
        self.default_box = self.default_box_fun.forward()

    def forward(self, x):
        '''
        input: 4d tensor, b, c, h, w
        output:
            default_box: (N, 4), vstack of [cx, cy, scale_w, scale_h]
            loc: (b, N, 4), b is batch index
            conf: (b, N, n_classes)
        '''
        input_feature_map = []
        loc= []
        conf = []

        # conv4_3 after relu
        for i in range(23):
            x = self.base[i](x)
        # x_norm = self.l2norm(x) # l2 norm in conv4_3
        x_norm = x
        input_feature_map.append(x_norm)

        for i in range(23, len(self.base)):
            x = self.base[i](x)
        input_feature_map.append(x) # conv7

        for i, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                input_feature_map.append(x)

        assert len(input_feature_map) == 6

        for in_feature, conf_layer, loc_layer in zip(input_feature_map, self.conf, self.loc):
            conf_out = conf_layer(in_feature) # b, c, h, w
            conf_out = conf_out.permute(0, 2, 3, 1).contiguous() # b, h, w, c
            conf.append(conf_out)
            loc_out = loc_layer(in_feature)
            loc_out = loc_out.permute(0, 2, 3, 1).contiguous()
            loc.append(loc_out)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        return self.default_box, loc, conf


def build_ssd(cfg):
    base = make_vgg(base_cfg[str(cfg.input_size)])
    extras = make_extras()
    head = make_head(base, extras, cfg.num_classes)

    return SSD(base, extras, head, cfg)


if __name__ == '__main__':
    model = build_ssd(21)
    inputs = torch.zeros((2, 3, 300, 300))
    print(inputs.shape)
    model(inputs)


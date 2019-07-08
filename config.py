class Config:
    data_root = '/Users/qiniu/data/VOCdevkit'
    train_set = [('2007', 'trainval')]
    val_set = [('2007', 'test')]

    num_classes = 21
    input_size = 300
    feature_maps = [38, 19, 10, 5, 3, 1]
    ratios = [[2], [3, 2], [3, 2], [3, 2], [2], [2]]
    min_scale = 0.2
    max_scale = 0.9
    variances = [1.0, 1.0] # target transform trick in ssd
    threshold = 0.5 # default box match iou threshold
    neg_pos_ratio = 3. # neg / pos default_box

    epochs = 1
    workers = 4
    batch_size = 1
    lr = 0.001
    weight_decay = 5e-4
    momentum = 0.9
    gpus = '0' # gpus to use, '0,1,2,3'
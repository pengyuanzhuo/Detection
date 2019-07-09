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
    lr = 0.00001
    lr_step = (80000, 100000, 120000) # lr * 0.1 for lr_step
    weight_decay = 5e-4
    momentum = 0.9
    gpus = None # gpus to use, '0,1,2,3', None for cpu
    pretrain = './weights/vgg16_reducedfc.pth' # backbone pretrain model. specifying checkpoint will override pretrain
    checkpoint = None # if resume
    checkpoint_dir = './checkpoints' # checkpoint dir
    print_freq = 1 # every print_freq
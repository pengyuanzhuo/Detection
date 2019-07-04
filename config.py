class Config:
    num_classes = 21
    input_size = 300
    feature_maps = [38, 19, 10, 5, 3, 1]
    ratios = [[2], [3, 2], [3, 2], [3, 2], [2], [2]]
    min_scale = 0.2
    max_scale = 0.9
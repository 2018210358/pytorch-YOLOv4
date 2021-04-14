import os
from easydict import EasyDict

# 训练数据形式
# image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
# image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...


hyparas = EasyDict()

base_dir = os.path.dirname(os.path.abspath(__file__))

hyparas.use_darknet_cfg = True
hyparas.cfgfile = os.path.join(base_dir, 'cfg', 'ghost-shuffle-large-yolov4-nose.cfg')

hyparas.batch = 64
hyparas.subdivisions = 4
hyparas.width = 608
hyparas.height = 608
hyparas.channels = 3
hyparas.momentum = 0.949
hyparas.decay = 0.0005
hyparas.angle = 0
hyparas.saturation = 1.5
hyparas.exposure = 1.5
hyparas.hue = .1

hyparas.learning_rate = 0.00261
hyparas.burn_in = 1000
hyparas.max_batches = 500500
hyparas.steps = [400000, 450000]
hyparas.policy = hyparas.steps
hyparas.scales = .1, .1

hyparas.cutmix = 0
hyparas.mosaic = 1

hyparas.letter_box = 0
hyparas.jitter = 0.2
hyparas.classes = 80
hyparas.track = 0
hyparas.w = hyparas.width
hyparas.h = hyparas.height
hyparas.flip = 1
hyparas.blur = 0
hyparas.gaussian = 0
hyparas.boxes = 60
hyparas.TRAIN_EPOCHS = 300
hyparas.train_label = os.path.join(base_dir, 'data', 'train.txt')
hyparas.val_label = os.path.join(base_dir, 'data', 'val.txt')
hyparas.TRAIN_OPTIMIZER = 'adam'

if hyparas.mosaic and hyparas.cutmix:
    hyparas.mixup = 4
elif hyparas.cutmix:
    hyparas.mixup = 2
elif hyparas.mosaic:
    hyparas.mixup = 3

hyparas.checkpoints = os.path.join(base_dir, 'checkpoints')
hyparas.TRAIN_TENSORBOARD_DIR = os.path.join(base_dir, 'log')

hyparas.iou_type = 'iou'

hyparas.keep_checkpoint_max = 10


# hyparas.learning_rate = 0.001
# hyparas.load = None
hyparas.gpu = '-1'
# hyparas.data_dir = None
# hyparas.pretrained = None
# hyparas.classes = 80
# hyparas.train_label_path = 'train.txt'
# hyparas.optimizer = 'adam'



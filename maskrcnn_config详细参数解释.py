"""
Mask R-CNN
基础配置类.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
#导入库
import math
import numpy as np


# 基础配置
# 不要直接使用它，而是进行子类化，将自己需要修改的地方覆盖代码相应的地方。
#定义一个基础配置的类。
class Config(object):
    """对于这个基础配置，是为了自定义配置，应用时可以创建一个子类继承它，并修改符合需要的属性。
    """
    # 下面是名称的配置，例如NAME = coco，当我们的代码在不同的实验中需要执行不同的操作时，这是很有用的。
    NAME = None  # 根据不同的实验要求修改None。

    # 使用的gpu的数量，当使用cpu训练时，数量为1
    GPU_COUNT = 1

    # 每个GPU上要训练的图像数。一个12 GB的GPU通常可以处理2张1024x1024像素的图像。
    # 根据您的GPU内存和图像大小进行调整。使用GPU能够处理的最高数字，以获得最佳性能
    IMAGES_PER_GPU = 2

    # steps_per_epoch为keras中的fit_generator函数中的参数，可以理解为每轮迭代多少次，fit_generator就是
    # 因为训练集太大而设置的自动生成batch的函数，当这轮生成的batch数达到steps_per_epoch就进行下一个epoch。
    # steps_per_epoch不需要考虑训练集大小，tensorboard会自动更新并在每个epoch结束时保存更新，此参数小意味着tensorboard更新的快。
    # 验证集也是在epoch最后记录的，而且比较花时间，所以steps_per_epoch不要设置太小，避免在验证上花大量时间。
    #按需求修改参数值
    STEPS_PER_EPOCH = 1000

    # validation_steps也是fit_generator中的参数，当validation_data为验证集生成器时，validation_steps同样是验证集一轮的迭代次数，
    # 即生成的batch数，此参数越大验证越精确，但同样训练的也越慢。
    # 在keras中，steps一般就是batchs of samples。
    #按需求修改参数值。
    VALIDATION_STEPS = 50

    # 主干网络结构，resnet50 or resner101
    BACKBONE = "resnet101"

    # 基于主干网络的降采样倍数，参照物是原始图片，这不能更改，在FPN中需要用到。
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 训练集中图片的类别数，包括背景
    NUM_CLASSES = 1  # 按要求修改

    # 使用FPN进行RPN时，从p2-p6对应的锚框的尺寸大小
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    #基于rpn_anchor_scales的每个尺寸又对应3种比例。
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # 锚步长
    # 如果为1就为feature map中的每个点创建锚，如果是2就在feature map上跨一个点创建锚，以此类推。
    RPN_ANCHOR_STRIDE = 1

    # 非极大值抑制的阈值，可以剔除rpn生成的proposals中不合格的，在训练时可以修改阈值。
    #注意NMS只在RPN网络的预测阶段在proposal layer使用
    RPN_NMS_THRESHOLD = 0.7

    # 在进行rpn训练时每幅图片使用多少个anchor
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # 进行非极大值抑制后在训练和测试阶段输入的ROI个数，即通过proposal layer之后的rois的个数
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # instance masks 使用小尺寸以节省内存，建议使用高分辨率图片
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # 输入图片尺寸重置
    #通常将图片设置为正方形，这样训练效果一般很好，重置大小是短边为IMAGE_MIN_DIM，长边不超过IMAGE_MAX_DIM，然后
    # 用0填充至正方形，多个图片可以设置为一个batch
    #具体方法:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    # 我们使用第二种方法
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # 最小缩放比率。在MIN_IMAGE_DIM后进行检查，并可以强制进一步缩放。
    # 例如，如果设置为2，那么图像的宽度和高度会增加一倍，甚至更多，
    # 即使MIN_IMAGE_DIM不需要它。但是，在“square”模式下，它可以被IMAGE_MAX_DIM覆盖。
    IMAGE_MIN_SCALE = 0

    # Image mean (RGB)，np.array创建一个数组
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 每副图像使用多少roi作为后面分类和掩膜网络的输入，论文中采用512，但通常
    # rpn产生不了那么多positive proposal,使得positive:negative为1:3
    # 但可以修改非极大值抑制的阈值来增加roi数量。注意这是detection target layer的输出
    TRAIN_ROIS_PER_IMAGE = 200 #train_rois_per_image

    # Positive roi的比例
    ROI_POSITIVE_RATIO = 0.33

    # poolalign中pooling后的大小，分为两段，一段用来分类，一段用来掩膜
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # 掩膜网络最后的输出，如果改变这个值，就需要修改对应的掩膜网络，
    MASK_SHAPE = [28, 28]

    # 在一副图像中最多有多少gt实例数目
    MAX_GT_INSTANCES = 100

    # 在rpn和最后分类中做bounding boxing 回归的标准偏差。
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # 最终检测包含的实例最大数量，即在测试阶段经过分类回归后的proposal中串行的输入到mask网络中的roi数量
    DETECTION_MAX_INSTANCES = 100

    # 在测试阶段的detection layer中用来过滤实例的分类得分没有达到这个置信度的目标，即在测试阶段的输入
    # 由RPN网络生成的rois，经过fast rcnn后输出每个roi的分类得分和第二次边界回归值，然后判别这个分类得分中最高的类的得分
    # 大于下面的置信度的roi才进入掩码网络生成掩码
    DETECTION_MIN_CONFIDENCE = 0.7 # 置信度小于0.7就跳过检测

    #  最终目标检测后被判别为属于实例的rois之间的NMS的非极大值抑制，这个是第二次修正后且置信度大于0.7的rois与GT的NMS阈值
    # 且经过NMS后只选择置信度前100名输入到下一个mask网络
    DETECTION_NMS_THRESHOLD = 0.3

    # 学习率和动量
    # 论文中学习率为0.02，会造成梯度爆炸，所以修改了
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # 正则化权重衰减系数
    WEIGHT_DECAY = 0.0001

    # 为了更好的优化设置的损失权重，在训练阶段可以使用
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    #在训练时是否使用rpn生成的roi，一般情况下为true，只有不经过rpn网络，直接调试后面的分类或者掩膜时，可以改为false
    USE_RPN_ROIS = True

    # 是否使用bn
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (不使用). Set layer in training mode even when inferencing
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # 梯度标准裁剪，暂时不知道用来干什么
    GRADIENT_CLIP_NORM = 5.0 # gradient_clip_norm

    def __init__(self):
        """设置计算属性值"""
        # batch size大小如何设置
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # 输入图像尺寸如何重置
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length，暂时不知道干啥用
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """在不同的实验中显示配置中对应属性的值"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

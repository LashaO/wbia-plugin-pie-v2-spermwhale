# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = "resnet50"
    cfg.model.pretrained = True  # load pretrained model weights if available
    cfg.model.load_weights = ""  # path to model weights
    cfg.model.resume = ""  # path to checkpoint for resume training
    cfg.model.use_aux = False
    cfg.model.n_parts = 1
    cfg.model.num_train_classes = 1

    # data
    cfg.data = CN()
    cfg.data.type = "image"
    cfg.data.root = "reid-data"
    cfg.data.source = "whaleshark"
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.height = 256  # image height
    cfg.data.width = 256  # image width
    cfg.data.transforms_train = ["random_flip"]  # training data augmentation
    cfg.data.transforms = ["resize"]  # depricated. Keep for consistency
    cfg.data.transforms_test = ["resize"]  # testing data augmentation
    cfg.data.k_tfm = 1  # number of times to apply augmentation to an image
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = "log"  # path to save log
    cfg.data.version = "v0"
    cfg.data.tb_dir = "tb_log"
    cfg.data.use_viewpoint = False
    cfg.data.normalize_viewpoint = False
    cfg.data.train_min_samples = 3
    cfg.data.test_min_samples = 2
    cfg.data.train_max_samples = None
    cfg.data.test_max_samples = None
    cfg.data.split_test = "val2021"
    cfg.data.viewpoint_list = None
    cfg.data.crop = False

    # data fields added for auto_train
    cfg.data.coco_dir = None  # coco-formatted training data directory
    cfg.data.split = None  # can specify split path for above (not required)
    cfg.data.split_test = None  # same as split but for test data
    cfg.data.dataset_url = None

    # sampler
    cfg.sampler = CN()
    cfg.sampler.num_instances = 4  # number of instances per identity
    cfg.sampler.num_copies = 1

    # train
    cfg.train = CN()
    cfg.train.optim = "adam"
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = [
        "classifier"
    ]  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ["classifier"]  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = "single_step"
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed
    cfg.train.eval_start = False

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.0  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = "softmax"
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.0  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.0  # weight to balance cross entropy loss
    cfg.loss.triplet.weight_c = 0.0005
    cfg.loss.weight_lab = 1.0
    cfg.loss.weight_unl = 1.0

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = "euclidean"  # metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # normalize feat vect before comp dist
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = -1  # evaluation frequency
    # (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results
    # (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visrank_resize = True  # if True resize images for visualization
    cfg.test.fliplr = False  # if True flip test image left-right
    cfg.test.fliplr_view = []  # viewpoint annotation to flip left-right

    return cfg


def imagedata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "source": cfg.data.source,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms_train": cfg.data.transforms_train,
        "transforms_test": cfg.data.transforms_test,
        "k_tfm": cfg.data.k_tfm,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "num_copies": cfg.sampler.num_copies,
    }


def optimizer_kwargs(cfg):
    return {
        "optim": cfg.train.optim,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "momentum": cfg.sgd.momentum,
        "sgd_dampening": cfg.sgd.dampening,
        "sgd_nesterov": cfg.sgd.nesterov,
        "rmsprop_alpha": cfg.rmsprop.alpha,
        "adam_beta1": cfg.adam.beta1,
        "adam_beta2": cfg.adam.beta2,
        "staged_lr": cfg.train.staged_lr,
        "new_layers": cfg.train.new_layers,
        "base_lr_mult": cfg.train.base_lr_mult,
    }


def lr_scheduler_kwargs(cfg):
    return {
        "lr_scheduler": cfg.train.lr_scheduler,
        "stepsize": cfg.train.stepsize,
        "gamma": cfg.train.gamma,
        "max_epoch": cfg.train.max_epoch,
    }


def engine_run_kwargs(cfg):
    return {
        "max_epoch": cfg.train.max_epoch,
        "start_epoch": cfg.train.start_epoch,
        "fixbase_epoch": cfg.train.fixbase_epoch,
        "open_layers": cfg.train.open_layers,
        "start_eval": cfg.test.start_eval,
        "eval_freq": cfg.test.eval_freq,
        "test_only": cfg.test.evaluate,
        "print_freq": cfg.train.print_freq,
        "dist_metric": cfg.test.dist_metric,
        "normalize_feature": cfg.test.normalize_feature,
        "visrank": cfg.test.visrank,
        "visrank_topk": cfg.test.visrank_topk,
        "ranks": cfg.test.ranks,
        "rerank": cfg.test.rerank,
        "visrank_resize": cfg.test.visrank_resize,
    }

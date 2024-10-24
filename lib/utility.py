import json
import os.path as osp
import time
import torch
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

# private package
from conf import *
from lib.dataset import AlignmentDataset,meDataset
from lib.backbone import StackedHGNetV1,Stacked3DHGNet
from lib.loss import *
from lib.metric import NME, FR_AUC
from lib.utils import convert_secs2time
from lib.utils import AverageMeter

torch_ver = torch.__version__.split('.')

def get_config(args):
    config = None
    config_name = args.config_name
    if config_name == "alignment":
        config = Alignment(args)
    else:
        assert NotImplementedError

    return config


def get_dataset(config, tsv_file, image_dir, loader_type, is_train):
    dataset = None
    if loader_type == "alignment":
        dataset = AlignmentDataset(
            tsv_file,
            image_dir,
            transforms.Compose([transforms.ToTensor()]),
            config.width,
            config.height,
            config.channels,
            config.means,
            config.scale,
            config.classes_num,
            config.crop_op,
            config.aug_prob,
            config.edge_info,
            config.flip_mapping,
            is_train,
            encoder_type=config.encoder_type
        )
    elif loader_type == "CASME2":
        dataset = meDataset(
            width=config.width,
            height=config.height,
            channels=config.channels,
            video_path=config.video_path,
            detector=config.detector,
            sp=config.sp,
        )
    else:
        assert False
    return dataset


def get_dataloader(config, data_type, world_rank=0, world_size=1):
    loader = None
    if data_type == "train":
        dataset = get_dataset(
            config,
            config.train_tsv_file,
            config.train_pic_dir,
            config.loader_type,
            is_train=True)
        if world_size > 1:
            sampler = DistributedSampler(dataset, rank=world_rank, num_replicas=world_size, shuffle=True)
            loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size // world_size,
                                num_workers=config.train_num_workers, pin_memory=True, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                num_workers=config.train_num_workers)
    elif data_type == "val":
        dataset = get_dataset(
            config,
            config.val_tsv_file,
            config.val_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.val_batch_size,
                            num_workers=config.val_num_workers)
    elif data_type == "test":
        dataset = get_dataset(
            config,
            config.test_tsv_file,
            config.test_pic_dir,
            config.loader_type,
            is_train=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,
                            num_workers=config.test_num_workers)
    elif data_type == "extract_feature":
        dataset = get_dataset(
            config,
            config.test_tsv_file,
            config.test_pic_dir,
            config.loader_type,
            is_train=False
        )
        loader = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size,
                            num_workers=config.test_num_workers)
    else:
        assert False
    return loader


def get_optimizer(config, net):
    params = net.parameters()

    optimizer = None
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.learn_rate)
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=config.learn_rate,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
    else:
        assert False
    return optimizer


def get_scheduler(config, optimizer):
    if config.scheduler == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    else:
        assert False
    return scheduler


def get_net(config):
    net = None
    if config.net == "stackedHGnet_v1":
        net = StackedHGNetV1(config=config,
                             classes_num=config.classes_num,
                             edge_info=config.edge_info,
                             nstack=config.nstack,
                             add_coord=config.add_coord,
                             decoder_type=config.decoder_type)

        # In future (torch 2.0), we may wish to compile the model but this currently doesn't work
        #if int(torch_ver[0]) >= 2:
        #    net = torch.compile(net)
    else:
        assert False
    return net


def get_criterions(config):
    criterions = list()
    for k in range(config.label_num):
        if config.criterions[k] == "AWingLoss":
            criterion = AWingLoss()
        elif config.criterions[k] == "smoothl1":
            criterion = SmoothL1Loss()
        elif config.criterions[k] == "l1":
            criterion = F.l1_loss
        elif config.criterions[k] == 'l2':
            criterion = F.mse_loss
        elif config.criterions[k] == "STARLoss":
            criterion = STARLoss(dist=config.star_dist, w=config.star_w)
        elif config.criterions[k] == "STARLoss_v2":
            criterion = STARLoss_v2(dist=config.star_dist, w=config.star_w)
        else:
            assert False
        criterions.append(criterion)
    return criterions


def set_environment(config):
    if config.device_id >= 0:
        assert torch.cuda.is_available() and torch.cuda.device_count() > config.device_id
        torch.cuda.empty_cache()
        config.device = torch.device("cuda", config.device_id)
        config.use_gpu = True
    else:
        config.device = torch.device("cpu")
        config.use_gpu = False

    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)  # ignore extremely small value
    torch.backends.cudnn.benchmark = True  # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.autograd.set_detect_anomaly(True)


def forward(config, test_loader, net):
    # ave_metrics = [[0, 0] for i in range(config.label_num)]
    list_nmes = [[] for i in range(config.label_num)]
    metric_nme = NME(nme_left_index=config.nme_left_index, nme_right_index=config.nme_right_index)
    metric_fr_auc = FR_AUC(data_definition=config.data_definition)

    output_pd = None

    net = net.float().to(config.device)
    net.eval()
    dataset_size = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    if config.logger is not None:
        config.logger.info("Forward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))
    for i, sample in enumerate(tqdm(test_loader)):
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels

        with torch.no_grad():
            output, heatmap, landmarks = net(input)

            # metrics
        for k in range(config.label_num):
            if config.metrics[k] is not None:
                list_nmes[k] += metric_nme.test(output[k], labels[k])

    metrics = [[np.mean(nmes), ] + metric_fr_auc.test(nmes) for nmes in list_nmes]

    return output_pd, metrics


def compute_loss(config, criterions, output, labels, heatmap=None, landmarks=None):
    batch_weight = 1.0
    sum_loss = 0
    losses = list()
    for k in range(config.label_num):
        if config.criterions[k] in ['smoothl1', 'l1', 'l2', 'WingLoss', 'AWingLoss']:
            loss = criterions[k](output[k], labels[k])
        elif config.criterions[k] in ["STARLoss", "STARLoss_v2"]:
            _k = int(k / 3) if config.use_AAM else k
            loss = criterions[k](heatmap[_k], labels[k])
        else:
            assert NotImplementedError
        loss = batch_weight * loss
        sum_loss += config.loss_weights[k] * loss
        loss = float(loss.data.cpu().item())
        losses.append(loss)
    return losses, sum_loss


def forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer, epoch):
    """
    Performs the forward and backward pass for training the network.
    
    Args:
        config: Configuration parameters, including training settings and logger.
        train_loader: DataLoader for the training dataset.
        net_module: The network module for the forward pass.
        net: The main network, could be different from net_module if using DataParallel.
        net_ema: The exponential moving average network, used for better generalization.
        criterions: A list of loss functions.
        optimizer: The optimizer for updating network parameters.
        epoch: The current epoch number.
    """
    # Initialize AverageMeter to record training time
    train_model_time = AverageMeter()
    # Initialize a list to record average losses
    ave_losses = [0] * config.label_num

    # Move the network module to the specified device and set it to training mode
    net_module = net_module.float().to(config.device)
    net_module.train(True)
    
    # Get the size of the dataset and batch size
    dataset_size = len(train_loader.dataset)
    batch_size = config.batch_size  # train_loader.batch_size
    batch_num = max(dataset_size / max(batch_size, 1), 1)
    
    # Log training information if a logger is configured
    if config.logger is not None:
        config.logger.info(config.note)
        config.logger.info("Forward Backward process, Dataset size: %d, Batch size: %d" % (dataset_size, batch_size))

    # Get the number of iterations
    iter_num = len(train_loader)
    epoch_start_time = time.time()
    # Set the epoch for the sampler if net_module and net are different
    if net_module != net:
        train_loader.sampler.set_epoch(epoch)
    
    # Iterate through the dataset
    for iter, sample in enumerate(train_loader):
        iter_start_time = time.time()
        
        # Prepare input data and labels
        input = sample["data"].float().to(config.device, non_blocking=True)
        labels = list()
        if isinstance(sample["label"], list):
            for label in sample["label"]:
                label = label.float().to(config.device, non_blocking=True)
                labels.append(label)
        else:
            label = sample["label"].float().to(config.device, non_blocking=True)
            for k in range(label.shape[1]):
                labels.append(label[:, k])
        labels = config.nstack * labels
        
        # Forward pass
        output, heatmaps, landmarks = net_module(input)

        # Calculate losses
        losses, sum_loss = compute_loss(config, criterions, output, labels, heatmaps, landmarks)
        ave_losses = list(map(sum, zip(ave_losses, losses)))

        # Backward pass and optimize
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            sum_loss.backward()
        optimizer.step()

        # Update the exponential moving average network
        if net_ema is not None:
            accumulate_net(net_ema, net, 0.5 ** (config.batch_size / 10000.0))

        # Record and display training information
        train_model_time.update(time.time() - iter_start_time)
        last_time = convert_secs2time(train_model_time.avg * (iter_num - iter - 1), True)
        if iter % config.display_iteration == 0 or iter + 1 == len(train_loader):
            if config.logger is not None:
                losses_str = ' Average Loss: {:.6f}'.format(sum(losses) / len(losses))
                for k, loss in enumerate(losses):
                    losses_str += ', L{}: {:.3f}'.format(k, loss)
                config.logger.info(
                    ' -->>[{:03d}/{:03d}][{:03d}/{:03d}]'.format(epoch, config.max_epoch, iter, iter_num) \
                    + last_time + losses_str)

    # Calculate total time and data loading time for the epoch
    epoch_end_time = time.time()
    epoch_total_time = epoch_end_time - epoch_start_time
    epoch_load_data_time = epoch_total_time - train_model_time.sum
    
    # Log time information for the epoch
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average total time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_total_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average loading data time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, epoch_load_data_time / iter_num))
        config.logger.info("Train/Epoch: %d/%d, Average training model time cost per iteration in this epoch: %.6f" % (
            epoch, config.max_epoch, train_model_time.avg))

    # Calculate and log average losses for the epoch
    ave_losses = [loss / iter_num for loss in ave_losses]
    if config.logger is not None:
        config.logger.info("Train/Epoch: %d/%d, Average Loss in this epoch: %.6f" % (
            epoch, config.max_epoch, sum(ave_losses) / len(ave_losses)))
    for k, ave_loss in enumerate(ave_losses):
        if config.logger is not None:
            config.logger.info("Train/Loss%03d in this epoch: %.6f" % (k, ave_loss))


def accumulate_net(model1, model2, decay):
    """
    使用指数加权平均方法合并两个模型的参数和缓冲区数据。
    
    此函数执行的操作为：model1 = model1 * decay + model2 * (1 - decay)，
    其中decay是一个用于控制两个模型参数加权的因子。
    
    参数:
    - model1: 第一个模型，其参数和缓冲区数据将被更新。
    - model2: 第二个模型，其参数和缓冲区数据用于更新model1。
    - decay: 衰减因子，用于控制两个模型参数的加权比例。
    
    返回值:
    此函数没有返回值，但它会直接更新model1的参数和缓冲区数据。
    """
    # 获取两个模型的参数字典
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    
    # 更新model1的参数，使用model1和model2的参数进行指数加权平均
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)
    
    # 获取两个模型的缓冲区字典
    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    
    # 更新model1的缓冲区数据，如果数据是浮点类型，则使用指数加权平均；
    # 否则，直接使用model2的缓冲区数据更新model1
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)


def save_model(config, epoch, net, net_ema, optimizer, scheduler, pytorch_model_path):
    # save pytorch model
    state = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }
    if config.ema:
        state["net_ema"] = net_ema.state_dict()

    torch.save(state, pytorch_model_path)
    if config.logger is not None:
        config.logger.info("Epoch: %d/%d, model saved in this epoch" % (epoch, config.max_epoch))

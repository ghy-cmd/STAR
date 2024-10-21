import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.coord_conv import CoordConvTh
from lib.dataset import get_decoder



class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        """
        初始化自定义层
        
        参数:
            kind (str): 指定激活和标准化类型
            channel (int): 输入数据的通道数
        """
        super().__init__()
        self.kind = kind

        # 分离标准化和激活类型
        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        # 初始化标准化函数
        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        # 初始化激活函数
        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'


class ConvBlock(nn.Module):
    """
    卷积块类，包含卷积层、批量归一化层和ReLU激活层的组合。
    
    参数:
    - inp_dim (int): 输入通道数。
    - out_dim (int): 输出通道数。
    - kernel_size (int, optional): 卷积核大小，默认为3。
    - stride (int, optional): 步长，默认为1。
    - bn (bool, optional): 是否包含批量归一化层，默认为False。
    - relu (bool, optional): 是否包含ReLU激活层，默认为True。
    - groups (int, optional): 分组卷积的组数，默认为1。
    """
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        # 初始化卷积层，使用同态padding保持输入输出尺寸相同
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size - 1) // 2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        # 根据参数relu决定是否初始化ReLU激活层
        if relu:
            self.relu = nn.ReLU()
        # 根据参数bn决定是否初始化批量归一化层
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """
    实现一个残差块（Residual Block），该块包含三个卷积层和一个捷径连接（skip connection）。
    残差块主要用于深度神经网络中，以缓解梯度消失问题并改善网络的训练效果。
    
    参数:
    - inp_dim: 输入的特征维度
    - out_dim: 输出的特征维度
    - mid_dim: 中间层的特征维度，默认为输出维度的一半
    """
    
    def __init__(self, inp_dim, out_dim, mid_dim=None):
        super(ResBlock, self).__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2  # 如果未指定中间维度，则设置为输出维度的一半
        self.relu = nn.ReLU()  # 使用ReLU激活函数
        self.bn1 = nn.BatchNorm2d(inp_dim)  # 第一个批归一化层，作用于输入特征
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)  # 第一个卷积块，减少特征维度
        self.bn2 = nn.BatchNorm2d(mid_dim)  # 第二个批归一化层，作用于中间维度特征
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)  # 第二个卷积块，保持中间维度
        self.bn3 = nn.BatchNorm2d(mid_dim)  # 第三个批归一化层，作用于中间维度特征
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)  # 第三个卷积块，增加特征维度到输出维度
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)  # 捷径层，用于调整输入特征维度以匹配输出
        if inp_dim == out_dim:
            self.need_skip = False  # 如果输入输出维度相同，则不需要捷径层
        else:
            self.need_skip = True  # 如果输入输出维度不同，则需要捷径层进行维度调整

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0, up_mode='nearest',
                 add_coord=False, first_one=False, x_dim=64, y_dim=64):
        """
        初始化Hourglass网络。

        参数:
        n -- Hourglass网络的层数
        f -- 基础特征通道数
        increase -- 每层递增的特征通道数（默认为0，不递增）
        up_mode -- 上采样模式（默认为'nearest'）
        add_coord -- 是否添加坐标信息到输入（默认为False）
        first_one -- 是否是第一个Hourglass模块（默认为False）
        x_dim -- 输入的x维度大小（默认为64）
        y_dim -- 输入的y维度大小（默认为64）
        """
        super(Hourglass, self).__init__()
        nf = f + increase

        Block = ResBlock

        if add_coord:
            # 使用CoordConvTh层替换普通卷积层，增加坐标位置信息
            self.coordconv = CoordConvTh(x_dim=x_dim, y_dim=y_dim,
                                        with_r=True, with_boundary=True,
                                        relu=False, bn=False,
                                        in_channels=f, out_channels=f,
                                        first_one=first_one,
                                        kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.coordconv = None
        # 上分支的第一个残差块
        self.up1 = Block(f, f)

        # 下分支的最大池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下分支的第一个残差块
        self.low1 = Block(f, nf)
        self.n = n
        # 递归的Hourglass模块或一个残差块（当n为1时）
        if self.n > 1:
            self.low2 = Hourglass(n=n - 1, f=nf, increase=increase, up_mode=up_mode, add_coord=False)
        else:
            self.low2 = Block(nf, nf)
        # 下分支的最后一个残差块
        self.low3 = Block(nf, f)
        # 上采样层
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        """
        初始化函数，构建边缘与关键点之间的关系矩阵，并为关键点未被边缘覆盖的情况添加偏置。

        :param edge_info: 边缘信息列表，每个元素包含一个布尔值表示边缘是否闭合，和一个关键点索引列表。
        :param num_points: 关键点的数量。
        :param num_edges: 边缘的数量。
        """
        super().__init__()

        # 初始化边缘到关键点的关系矩阵，大小为关键点数乘以边缘数。
        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        # 将关系矩阵注册为模型的缓冲区权重。
        # pn x en x 1 x 1.
        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        # 一些关键点可能没有被任何边缘覆盖，在这些情况下，我们需要在它们的热图权重上添加一个常数偏置。
        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        # pn x 1.
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        # input: batch_size x en x hw x hh.
        # output: batch_size x pn x hw x hh.
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class StackedHGNetV1(nn.Module):
    def __init__(self, config, classes_num, edge_info,
             nstack=4, nlevels=4, in_channel=256, increase=0,
             add_coord=True, decoder_type='default'):
        """
        初始化StackedHGNetV1模型。

        参数:
        - config: 配置对象，包含模型的配置信息。
        - classes_num: 包含模型类别数量的列表，分别对应热图、边图和点图的数量。
        - edge_info: 描述边的信息，用于边到热图的转换。
        - nstack: Hourglass模块的堆叠数量。
        - nlevels: Hourglass模块的层数。
        - in_channel: 输入通道数。
        - increase: 每个Hourglass模块的通道增加量。
        - add_coord: 是否添加坐标信息到输入中。
        - decoder_type: 解码器的类型。

        返回:
        无返回值。
        """
        super(StackedHGNetV1, self).__init__()

        # 模型配置和解码器类型
        self.cfg = config
        self.coder_type = decoder_type
        self.decoder = get_decoder(decoder_type=decoder_type)
        self.nstack = nstack
        self.add_coord = add_coord

        # 类别数量
        self.num_heats = classes_num[0]

        # 初始化卷积块，根据是否添加坐标信息选择不同的卷积块
        if self.add_coord:
            convBlock = CoordConvTh(x_dim=self.cfg.width, y_dim=self.cfg.height,
                                    with_r=True, with_boundary=False,
                                    relu=True, bn=True,
                                    in_channels=3, out_channels=64,
                                    kernel_size=7,
                                    stride=2, padding=3)
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)

        # 最大池化层
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义残差块
        Block = ResBlock

        # 前处理层
        self.pre = nn.Sequential(
            convBlock,
            Block(64, 128),
            pool,
            Block(128, 128),
            Block(128, in_channel)
        )

        # 堆叠的Hourglass模块
        self.hgs = nn.ModuleList(
            [Hourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, first_one=(_ == 0),
                    x_dim=int(self.cfg.width / self.nstack), y_dim=int(self.cfg.height / self.nstack))
            for _ in range(nstack)])

        # 特征处理模块
        self.features = nn.ModuleList([
            nn.Sequential(
                Block(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        # 输出热图模块
        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False)
            for _ in range(nstack)])

        # 如果配置中使用AAM，则初始化相关模块
        if self.cfg.use_AAM:
            self.num_edges = classes_num[1]
            self.num_points = classes_num[2]

            self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)
            self.out_edgemaps = nn.ModuleList(
                [ConvBlock(in_channel, self.num_edges, 1, relu=False, bn=False)
                for _ in range(nstack)])
            self.out_pointmaps = nn.ModuleList(
                [ConvBlock(in_channel, self.num_points, 1, relu=False, bn=False)
                for _ in range(nstack)])
            self.merge_edgemaps = nn.ModuleList(
                [ConvBlock(self.num_edges, in_channel, 1, relu=False, bn=False)
                for _ in range(nstack - 1)])
            self.merge_pointmaps = nn.ModuleList(
                [ConvBlock(self.num_points, in_channel, 1, relu=False, bn=False)
                for _ in range(nstack - 1)])
            self.edgemap_act = Activation("sigmoid", self.num_edges)
            self.pointmap_act = Activation("sigmoid", self.num_points)

        # 特征、热图的合并模块
        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
            for _ in range(nstack - 1)])
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
            for _ in range(nstack - 1)])

        # 模型的堆叠数量
        self.nstack = nstack

        # 热图激活函数
        self.heatmap_act = Activation("in+relu", self.num_heats)

        # 推理标志
        self.inference = False

    def set_inference(self, inference):
        self.inference = inference

    def forward(self, x):
        """
        网络的前向传播函数.

        参数:
            x: 输入张量, 代表输入图像.

        返回:
            y: 包含所有阶段输出的列表, 包括关键点热力图和, 如果使用AAM, 则还包括点图和边图.
            fusionmaps: 所有阶段的融合热力图.
            landmarks: 最终阶段的关键点坐标.
        """
        # 对输入进行预处理
        x = self.pre(x)

        # 初始化用于存储输出的列表或变量
        y, fusionmaps = [], []
        heatmaps = None
        for i in range(self.nstack):
            # 通过第i个堆栈的小时钟网络
            hg = self.hgs[i](x, heatmap=heatmaps)
            # 提取特征
            feature = self.features[i](hg)

            # 生成原始热力图
            heatmaps0 = self.out_heatmaps[i](feature)
            # 激活热力图
            heatmaps = self.heatmap_act(heatmaps0)

            # 根据配置决定是否使用AAM
            if self.cfg.use_AAM:
                # 生成点图、边图并进行激活
                pointmaps0 = self.out_pointmaps[i](feature)
                pointmaps = self.pointmap_act(pointmaps0)
                edgemaps0 = self.out_edgemaps[i](feature)
                edgemaps = self.edgemap_act(edgemaps0)
                # 计算融合热力图
                mask = self.e2h_transform(edgemaps) * pointmaps
                fusion_heatmaps = mask * heatmaps
            else:
                fusion_heatmaps = heatmaps

            # 从热力图中提取关键点
            landmarks = self.decoder.get_coords_from_heatmap(fusion_heatmaps)

            # 如果不是最后一个堆栈, 对x进行更新以供下一个堆栈使用
            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps)
                if self.cfg.use_AAM:
                    x += self.merge_pointmaps[i](pointmaps)
                    x += self.merge_edgemaps[i](edgemaps)

            # 将当前阶段的输出添加到列表中
            y.append(landmarks)
            if self.cfg.use_AAM:
                y.append(pointmaps)
                y.append(edgemaps)

            # 将当前阶段的融合热力图添加到列表中
            fusionmaps.append(fusion_heatmaps)

        # 返回所有阶段的输出、融合热力图和最终阶段的关键点坐标
        # return y, fusionmaps, landmarks
        return feature

class Stacked3DHGNet(nn.Module):
    def __init__(self,config, classes_num, edge_info,
             nstack=4, nlevels=4, in_channel=256, increase=0,
             add_coord=True, decoder_type='default', chunk_size=16):
        super(Stacked3DHGNet, self).__init__()
        self.stacked_hg_nets = nn.ModuleList([
            StackedHGNetV1(config, classes_num, edge_info,
                          nstack=nstack, nlevels=nlevels, in_channel=in_channel, increase=increase,
                          add_coord=add_coord, decoder_type=decoder_type).load_state_dict(torch.load(f'/home/Data/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl'))
            for _ in range(chunk_size)
        ])
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, 128),  # 假设 channels, height, width 是已知的
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出维度为2，分别表示起始点和结束点
        )
        # 分类头
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, 128),
            nn.ReLU(),
            nn.Linear(128, classes_num)  # 输出维度为类别数
        )
    
    def forward(self, x):
        # x 的维度为 (batch_size, chunk_size, channels, height, width)
        batch_size, chunk_size, channels, height, width = x.size()
        
        # 将 x 分解为 chunk_size 个 (batch_size, channels, height, width) 的张量
        x_chunks = torch.unbind(x, dim=1)
        
        # 并联处理每个 chunk
        outputs = [net(chunk) for net, chunk in zip(self.stacked_hg_nets, x_chunks)]
        
        # 将输出重新组合为 (batch_size, chunk_size, channels, height, width)
        final_output = torch.stack(outputs, dim=1)
        # 展平特征图并进行回归
        flat_output = final_output.view(batch_size, chunk_size, -1)
        regression_output = self.regression_head(flat_output)
        
        return regression_output
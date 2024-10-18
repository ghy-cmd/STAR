import torch
import torch.nn as nn


class AddCoordsTh(nn.Module):
    def __init__(self, x_dim, y_dim, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        网络的前向传播函数。
        
        参数:
        - input_tensor: 输入张量，形状为 (batch, c, x_dim, y_dim)。
        - heatmap: 可选参数，热力图张量，用于边界检测。

        返回值:
        - 经过输入张量和位置信息（以及边界信息）融合后的张量。
        """
        # 获取batch size
        batch_size_tensor = input_tensor.shape[0]

        # 创建包含y维度的ones张量
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).to(input_tensor) #(1, self.y_dim)
        xx_ones = xx_ones.unsqueeze(-1) #(1, self.y_dim, 1)

        # 创建包含x维度范围的张量
        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor) # (1,self.x_dim)
        xx_range = xx_range.unsqueeze(1) #(1, 1, self.x_dim)

        # 生成xx_channel张量，表示x轴的坐标信息
        xx_channel = torch.matmul(xx_ones.float(), xx_range.float()) #(1, self.y_dim, self.x_dim)
        xx_channel = xx_channel.unsqueeze(-1) #(1, self.y_dim, self.x_dim, 1)

        # 创建包含x维度的ones张量
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).to(input_tensor) #(1, self.x_dim)
        yy_ones = yy_ones.unsqueeze(1) #(1, 1, self.x_dim)

        # 创建包含y维度范围的张量
        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor) #(1, self.y_dim)
        yy_range = yy_range.unsqueeze(-1) #(1, self.y_dim, 1)

        # 生成yy_channel张量，表示y轴的坐标信息
        yy_channel = torch.matmul(yy_range.float(), yy_ones.float()) #(1, self.x_dim, self.y_dim)
        yy_channel = yy_channel.unsqueeze(-1) #(1, self.x_dim, self.y_dim, 1)

        # 调整xx_channel和yy_channel的维度顺序
        xx_channel = xx_channel.permute(0, 3, 2, 1) #(1, 1, self.x_dim, self.y_dim)
        yy_channel = yy_channel.permute(0, 3, 2, 1) #(1, 1, self.y_dim, self.x_dim)

        # 归一化xx_channel和yy_channel到范围[-1, 1]
        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # 复制xx_channel和yy_channel以匹配batch size
        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        # 如果启用边界检测并且heatmap不为空，则处理边界信息
        if self.with_boundary and heatmap is not None:
            # 获取边界通道并进行clamp操作限制值在[0.0, 1.0]之间
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

            # 创建全零张量
            zero_tensor = torch.zeros_like(xx_channel).to(xx_channel)
            # 根据边界通道筛选xx_channel和yy_channel，保留边界内的坐标信息
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)

        # 将输入张量和坐标信息张量（xx_channel和yy_channel）沿着通道维度合并
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1) #(batch_size, channels + 1 + 1, height, width)

        # 如果启用半径信息计算
        if self.with_r:
            # 计算每个点到中心点的距离，然后归一化
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            # 将半径信息rr与之前的张量沿着通道维度合并
            ret = torch.cat([ret, rr], dim=1)

        # 如果启用边界检测并且heatmap不为空，则将边界信息与之前的张量沿着通道维度合并
        if self.with_boundary and heatmap is not None:
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)

        # 返回最终的张量
        return ret


class CoordConvTh(nn.Module):
    """
    带坐标信息的卷积层类

    该类通过在输入特征图中添加坐标信息，增强模型对绝对位置的感知能力。可以选
    择是否添加边界信息，以及是否对输出应用ReLU激活函数和批量归一化。

    参数:
    - x_dim: int, x坐标的维度
    - y_dim: int, y坐标的维度
    - with_r: bool, 是否添加极坐标r（距离）
    - with_boundary: bool, 是否添加边界坐标信息
    - in_channels: int, 输入通道数（不包含坐标信息）
    - out_channels: int, 输出通道数
    - first_one: bool, 默认False。若为True，则不添加边界信息
    - relu: bool, 默认False。若为True，则在卷积后添加ReLU激活函数
    - bn: bool, 默认False。若为True，则在卷积后添加批量归一化层
    - *args, **kwargs: 其他参数，传递给nn.Conv2d

    属性:
    - addcoords: AddCoordsTh实例，用于添加坐标信息
    - conv: nn.Conv2d实例，执行卷积操作
    - relu: nn.ReLU实例或None，用于激活输出
    - bn: nn.BatchNorm2d实例或None，用于批量归一化输出
    - with_boundary: 保存with_boundary参数值
    - first_one: 保存first_one参数值
    """

    def __init__(self, x_dim, y_dim, with_r, with_boundary,
                 in_channels, out_channels, first_one=False, relu=False, bn=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        # 初始化坐标添加层
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r,
                                    with_boundary=with_boundary)
        # 根据配置更新输入通道数，以包含坐标信息
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        # 初始化卷积层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)
        # 根据配置初始化ReLU和批量归一化层
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        # 保存边界信息配置和first_one配置
        self.with_boundary = with_boundary
        self.first_one = first_one


    def forward(self, input_tensor, heatmap=None):
        """
        模型的前向传播函数。

        参数:
        - input_tensor: 输入张量，包含特征信息。
        - heatmap: 可选参数，边界信息，当with_boundary为True且不是第一个模块时必须提供。

        返回:
        - 经过坐标添加、卷积、批量归一化（可选）和激活（可选）后的输出张量。

        注释:
        - with_boundary和first_one属性用于控制是否使用边界信息。
        - addcoords函数用于将坐标和可能的边界信息添加到输入张量上。
        - conv执行卷积操作。
        - 如果初始化时选择了批量归一化（bn），则应用批量归一化。
        - 如果初始化时选择了激活函数（relu），则应用激活函数。
        """
        # 确保当模型需要边界信息且不是第一个模块时，heatmap必须被提供
        assert (self.with_boundary and not self.first_one) == (heatmap is not None)
        # 将坐标和可能的边界信息添加到输入张量上
        ret = self.addcoords(input_tensor, heatmap)
        # 执行卷积操作
        ret = self.conv(ret)
        # 如果初始化时选择了批量归一化，则应用批量归一化
        if self.bn is not None:
            ret = self.bn(ret)
        # 如果初始化时选择了激活函数，则应用激活函数
        if self.relu is not None:
            ret = self.relu(ret)

        return ret


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1).to(input_tensor)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).to(input_tensor)

        xx_channel = xx_channel / (x_dim - 1)
        yy_channel = yy_channel / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

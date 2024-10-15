import os
import cv2
import copy
import dlib
import math
import argparse
import numpy as np
import gradio as gr
from matplotlib import pyplot as plt
import torch
# private package
from lib import utility


class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        """
        构建旋转和缩放变换矩阵。
        
        该函数根据给定的旋转角度、缩放比例和位移，计算出一个变换矩阵，
        该矩阵用于将图像从一个中心点缩放并旋转到另一个中心点。
        
        参数:
        angle: float, 旋转角度，以弧度表示。
        scale: float, 缩放比例。
        shift_xy: tuple, 在x和y方向上的平移量。
        from_center: tuple, 图像旋转和缩放前的中心点坐标。
        to_center: tuple, 图像旋转和缩放后的中心点坐标。
        
        返回:
        rot_scale_m: numpy数组, 3x3的变换矩阵，将原图的中心点变换到目标中心点。
        """
        # 计算给定角度的余弦和正弦值，用于旋转计算
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        # 解包中心点坐标
        fx, fy = from_center
        tx, ty = to_center

        # 计算缩放后的余弦和正弦值
        acos = scale * cosv
        asin = scale * sinv

        # 计算变换矩阵的第一行
        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        # 计算变换矩阵的第二行
        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        # 构建变换矩阵
        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        
        # 返回变换矩阵
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        """
        根据给定的缩放比例和中心点信息，计算图像处理的转换矩阵。

        此函数主要用于计算将图像从一个中心点和缩放比例转换到另一个指定中心点和尺寸的过程。
        它涉及到图像的旋转、缩放和平移处理，是图像变换中的核心部分。

        参数:
        scale (float): 当前图像的缩放比例。
        center_w (float): 当前图像中心点的x坐标。
        center_h (float): 当前图像中心点的y坐标。

        返回:
        matrix (ndarray): 转换矩阵，用于将图像从当前状态转换到目标状态。
        """
        # 根据align_corners属性决定目标图像的宽度和高度，这个参数主要用于控制插值过程中如何映射源图像的角点到目标图像的角点。
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        # 初始化旋转和缩放参数
        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)

        # 计算并返回转换矩阵
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """

    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)


class Alignment:
    def __init__(self, args, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework

        # model
        if self.dl_framework == "pytorch":
            # conf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            # set environment
            utility.set_environment(self.config)
            self.config.init_instance()
            if self.config.logger is not None:
                self.config.logger.info("Loaded configure file %s: %s" % (args.config_name, self.config.id))
                self.config.logger.info("\n" + "\n".join(["%s: %s" % item for item in self.config.__dict__.items()]))

            net = utility.get_net(self.config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])
            net = net.to(self.config.device_id)
            net.eval()
            self.alignment = net
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale,
                                           align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        """
        将点从标准化坐标转换为图像坐标系。
        
        标准化坐标将图像看作一个边长为2的正方形，左下角为 (-1, -1)，右上角为 (1, 1)。
        这个函数将这样的坐标转换回图像的实际像素坐标。
        
        参数:
        - points: Tensor，形状为 (N, M, 2)，其中 N 是点集数量，M 是每个点集中的点数量，每个点是一个 (x, y) 二元组。
        - align_corners: Boolean，指定是否将输入坐标看作是像素的角点。如果为 True，输入坐标 (-1, -1) 对应图像左下角像素的角点，
        (1, 1) 对应图像右上角像素的角点。如果为 False，输入坐标 (-1, -1) 对应图像左下角像素的中心，(1, 1) 对应图像右上角像素的中心。
        
        返回值:
        - Tensor，形状同输入 points 相同，表示转换后的点的坐标。
        """
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            # 当 align_corners 为 True 时，将点从标准化坐标转换为以像素角点为参考的图像坐标。
            return (points + 1) / 2 * torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            # 当 align_corners 为 False 时，将点从标准化坐标转换为以像素中心为参考的图像坐标。
            return ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    def preprocess(self, image, scale, center_w, center_h):
        """
        对输入图像进行预处理，包括裁剪、变换和归一化，以适应后续的模型处理。

        参数:
            image: 待处理的原始图像。
            scale: 图像缩放比例，用于调整图像大小。
            center_w: 图像宽度方向的中心点坐标。
            center_h: 图像高度方向的中心点坐标。

        返回:
            input_tensor: 预处理后的图像张量，准备输入模型。
            matrix: 裁剪矩阵，用于将原始图像裁剪到以(center_w, center_h)为中心，scale为缩放比例的区域。
        """
        # 计算裁剪矩阵，用于后续的图像裁剪
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        
        # 应用裁剪矩阵，对原始图像进行裁剪和透视变换
        input_tensor = self.transformPerspective.process(image, matrix)
        
        # 增加一个维度，使input_tensor变为[N, H, W, C]的形状，适应模型输入要求
        input_tensor = input_tensor[np.newaxis, :]
        
        # 将numpy数组转换为PyTorch张量，并调整数据类型和维度顺序
        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        
        # 对图像张量进行归一化，将其值缩放到-1.0到1.0的范围之间
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        
        # 将张量移动到指定的设备（如GPU）上
        input_tensor = input_tensor.to(self.config.device_id)
        
        # 返回预处理后的图像张量和裁剪矩阵
        return input_tensor, matrix

    def postprocess(self, srcPoints, coeff):
        # dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        # matrix^(-1) * src = dst
        # src = matrix * dst
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, scale, center_w, center_h):
        input_tensor, matrix = self.preprocess(image, scale, center_w, center_h)

        if self.dl_framework == "pytorch":
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[-1][0]
            print(landmarks)
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix)) #逆矩阵变换坐标

        return landmarks


def draw_pts(img, pts, mode="pts", shift=4, color=(0, 255, 0), radius=1, thickness=1, save_path=None, dif=0,
             scale=0.3, concat=False, ):
    """
    在图像上绘制点或索引。

    :param img: 原始图像，可以是单通道或三通道图像。
    :param pts: 点的列表，每个点是一个包含x和y坐标的列表。
    :param mode: 绘制模式，可以是"index"（绘制点的索引）或"pts"（绘制点）。
    :param shift: 缩放因子，用于将点的坐标缩放到图像的实际尺寸。
    :param color: 绘制的颜色，对于RGB图像，默认为绿色。
    :param radius: 绘制点时的半径。
    :param thickness: 绘制文本或点的线宽。
    :param save_path: 保存绘制后图像的路径，如果为None，则不保存。
    :param dif: 点坐标的微调值。
    :param scale: 绘制文本时的缩放比例。
    :param concat: 是否将原始图像和绘制后的图像拼接在一起。
    :return: 绘制点后的图像。
    """
    # 深拷贝图像，以避免修改原始图像
    img_draw = copy.deepcopy(img)
    for cnt, p in enumerate(pts):
        if mode == "index":
            # 在图像上绘制点的索引
            cv2.putText(img_draw, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)
        elif mode == 'pts':
            # 绘制点，首先进行颜色空间的转换以修复OpenCV的BUG
            if len(img_draw.shape) > 2:
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            # 在图像上绘制点
            cv2.circle(img_draw, (int(p[0] * (1 << shift)), int(p[1] * (1 << shift))), radius << shift, color, -1,
                       cv2.LINE_AA, shift=shift)
        else:
            # 如果模式不是"index"或"pts"，抛出异常
            raise NotImplementedError
    # 按需拼接原始图像和绘制后的图像
    if concat:
        img_draw = np.concatenate((img, img_draw), axis=1)
    # 按需保存绘制后的图像
    if save_path is not None:
        cv2.imwrite(save_path, img_draw)
    return img_draw


def process(input_image):
    image_draw = copy.deepcopy(input_image)
    dets = detector(input_image, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        exit()

    results = []
    for detection in dets:
        face = sp(input_image, detection)
        shape = []
        for i in range(68):
            x = face.part(i).x
            y = face.part(i).y
            shape.append((x, y))
        shape = np.array(shape)
        # image_draw = draw_pts(image_draw, shape)
        x1, x2 = shape[:, 0].min(), shape[:, 0].max()
        y1, y2 = shape[:, 1].min(), shape[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05 # scale ratio
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2

        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        landmarks_pv = alignment.analyze(input_image, scale, center_w, center_h)
        results.append(landmarks_pv)
        image_draw = draw_pts(image_draw, landmarks_pv)
    return image_draw, results


if __name__ == '__main__':
    # face detector
    # could be downloaded in this repo: https://github.com/italojs/facial-landmarks-recognition/tree/master
    predictor_path = '/home/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    # facial landmark detector
    args = argparse.Namespace()
    args.config_name = 'alignment'
    args.data_definition = '300W'
    # could be downloaded here: https://drive.google.com/file/d/1aOx0wYEZUfBndYy_8IYszLPG_D2fhxrT/view
    model_path = '/home/Data/300W_STARLoss_NME_2_87.pkl'
    device_ids = '7'
    device_ids = list(map(int, device_ids.split(",")))
    alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=device_ids)

    # image:      input image
    # image_draw: draw the detected facial landmarks on image
    # results:    a list of detected facial landmarks
    face_file_path = '/home/Data/rawpic/s15/15_0101disgustingteeth/img001.jpg'
    image = cv2.imread(face_file_path)
    image_draw, results = process(image)

    # visualize
    img = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    cv2.imwrite('output.png', img[:, :, ::-1])  # 将 RGB 转回 BGR 格式再保存
    # demo
    # interface = gr.Interface(fn=process, inputs="image", outputs="image")
    # interface.launch(share=True)

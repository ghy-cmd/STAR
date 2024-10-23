import math
import os
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class meDataset(Dataset):

    def __init__(self,  width=256, height=256, channels=3,video_path=None, detector=None, sp=None):
        super(meDataset, self).__init__()

        self.image_width = width
        self.image_height = height
        self.channels = channels
        assert self.image_width == self.image_height
        self.video_path = video_path
        self.images = [i for i in os.listdir(video_path) if i.endswith('.jpg')]
        self.images= [int(str(i).split('.')[0][3:]) for i in self.images]
        self.images.sort()
        self.images = ['img' + str(i).zfill(3) + '.jpg' for i in self.images]
        self.images = [os.path.join(video_path, i) for i in self.images]
        self.scale, self.center_w, self.center_h = self._set_rotate_and_scale(self.images[0], detector, sp)
        self.matrix=self.get_crop_matrix(self.scale, self.center_w, self.center_h, align_corners=True)

        
    def _set_rotate_and_scale(self, image, detector, sp):
        image = cv2.imread(image, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
        assert image is not None and len(image.shape) == 3 and image.shape[2] == 3
        dets = detector(image, 1)

        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(self.video_path))
            exit()
        elif num_faces > 1:
            print("Sorry, there were more than one faces found in '{}'".format(self.video_path))
            exit()
            
        face = sp(image, dets[0])
        shape = []
        for i in range(68):
            x = face.part(i).x
            y = face.part(i).y
            shape.append((x, y))
        shape = np.array(shape)
        x1, x2 = shape[:, 0].min(), shape[:, 0].max()
        y1, y2 = shape[:, 1].min(), shape[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05 # scale ratio
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2
        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        return scale, center_w, center_h

    def get_crop_matrix(self, scale, center_w, center_h,align_corners):
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
        if align_corners:
            to_w, to_h = self.image_width - 1, self.image_width - 1
        else:
            to_w, to_h = self.image_width, self.image_width

        # 初始化旋转和缩放参数
        rot_mu = 0
        scale_mu = self.image_width / (scale * 200.0)
        shift_xy_mu = (0, 0)

        # 计算并返回转换矩阵
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix
    
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
    
    def processPerspective(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_width, self.image_width),
            flags=cv2.INTER_LINEAR, borderValue=0)
    
    def _load_image(self, image_path):
        if not os.path.exists(image_path):
            return None

        try:
            # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)#HWC, BGR, [0-255]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
            assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
        except:
            try:
                img = imageio.imread(image_path)  # HWC, RGB, [0-255]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
            except:
                try:
                    gifImg = imageio.mimread(image_path)  # BHWC, RGB, [0-255]
                    img = gifImg[0]  # HWC, RGB, [0-255]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                    assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
                except:
                    img = None
        return img



    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = dict()
        image_path=self.images[index]
        # image path
        sample["image_path"] = image_path

        img = self._load_image(image_path)  # HWC, BGR, [0, 255]
        assert img is not None
        img=self.processPerspective(img,self.matrix)
        img=torch.from_numpy(img)
        img=img.float().permute(2, 0, 1)
        img=img / 255.0 * 2.0 - 1.0
        sample["data"] = img  # CHW, BGR, [-1, 1]
        return sample
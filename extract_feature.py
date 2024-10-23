import argparse
import os
import dlib
import torch
from tqdm import tqdm
# private package
from lib import utility
import pandas as pd
import numpy as np

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
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix)) #逆矩阵变换坐标

        return landmarks

if __name__ == '__main__':
    # 设置 CUDA_VISIBLE_DEVICES 环境变量，使只有设备 7 可见
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # 检查当前设备
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device}")
    args = argparse.Namespace()
    args.config_name = 'alignment'
    args.data_definition = 'WFLW'
    args.device_ids = '7'
    args.device_ids = list(map(int, args.device_ids.split(",")))
    args.loader_type = 'CASME2'
    model_path = '/home/Data/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl'
    annotation_path = '/home/casme2_annotation.csv'
    predictor_path = '/home/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat'
    save_path = '/home/Data/casme2_face_feature'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    # conf
    config = utility.get_config(args)
    config.device_id = args.device_ids[0]

    # set environment
    utility.set_environment(config)
    config.init_instance()
    if config.logger is not None:
        config.logger.info("Loaded configure file %s: %s" % (args.config_name, config.id))
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))
    net = utility.get_net(config)
    if args.device_ids == [-1]:
        checkpoint = torch.load(model_path, map_location="cpu")
    else:
        checkpoint = torch.load(model_path)
        
    net.load_state_dict(checkpoint["net"])
    net.to(config.device_id)
    net.eval()

    if config.logger is not None:
        config.logger.info("Loaded network")
        # config.logger.info('Net flops: {} G, params: {} MB'.format(flops/1e9, params/1e6))
    config.detector=detector
    config.sp = sp
    df= pd.read_csv(annotation_path)
    # 转换为列表并剔除重复值
    all_video = list(set(df.iloc[:, 1].values))
    video_num=len(all_video)
    for i, video in enumerate(all_video):
        # config.logger.info("Processing {}/{} video: {}".format(i+1,video_num,video))
        config.video_path = video
        data_loader = utility.get_dataloader(config, "extract_feature")
        dataset_size = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        full_features = []
        for i, sample in enumerate(tqdm(data_loader,desc="Processing {}/{} video: {}".format(i+1,video_num,video))):
            input = sample["data"].float().to(config.device_id, non_blocking=True)
            with torch.no_grad():
                output, heatmap, landmarks, feature = net(input)
            full_features.append(feature.data.cpu().numpy())
        final_features = np.concatenate(full_features, axis=0)
        del full_features
        video_name=video.split('/')[-1]
        video_save_path=os.path.join(save_path, video_name)
        np.savez_compressed(video_save_path, feature=final_features, allow_pickle=False, fix_imports=True)
        
        

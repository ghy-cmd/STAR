import os
import pickle
from get_sliding_windows import video_process
from construct_feature import construct_feature
import argparse
import pandas as pd
import numpy as np

def cas_train_test(ann_path,
                    sample_interval, wind_length, window_sliding,
                    label_frequency,
                    is_train):
    """
    处理CASIA数据集以生成训练和测试数据。

    参数:
        ann_path (str): 注解文件的路径。
        info_dir_split (str): 保存分割信息的目录。
        info_dir_feature (str): 保存特征信息的目录。
        sample_interval (int): 采样帧的间隔。
        wind_length (int): 处理窗口的长度。
        window_sliding (int): 滑动窗口的步长。
        tem_feature_dir (str): 保存时间特征的目录。
        spa_feature_dir (str): 保存空间特征的目录。
        label_frequency (float): 标签过滤的频率阈值。
        is_train (bool): 是否为训练标志。
    """
    
    # 初始化主体列表和标签列表
    ca_subject = [15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40]
    ca_label = [i for i in range(1, len(ca_subject) + 1)]
    ca_subject = ['s' + str(i) for i in ca_subject]
    
    # 遍历每个主体以生成训练和测试数据
    for i in range(len(ca_subject)):
        ca_subject_test = ca_subject[i]
        ca_lebel_test = ca_label[i]
        ann_df = pd.read_csv(ann_path)
        ann_df_train = pd.read_csv(ann_path)
        ann_df_test = pd.read_csv(ann_path)
        # 将数据集拆分为训练集和测试集
        for i in range(len(ann_df)):
            if int(ann_df.subject.values[i]) == ca_lebel_test:
                ann_df_train.drop([i], inplace=True)
            else:
                ann_df_test.drop([i], inplace=True)
        
        # 训练窗口滑动
        gt_label, gt_box, gt_windows = video_process(ann_df_train, window_sliding, label_frequency, neg=True)
        # 保存处理后的训练数据
        with open(os.path.join(info_dir_split, 'gt_label_{}.pkl'.format(ca_subject_test)), 'wb') as f:
            pickle.dump(gt_label, f)
        with open(os.path.join(info_dir_split, 'gt_box_{}.pkl'.format(ca_subject_test)), 'wb') as f:
            pickle.dump(gt_box, f)
        with open(os.path.join(info_dir_split, 'window_info_train_{}.log'.format(ca_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows)
        
        # 处理训练特征
        feature_process(ca_subject_test,
                        gt_label, gt_box, 
                        gt_windows, info_dir_feature,
                        sample_interval, wind_length, tem_feature_dir,
                        spa_feature_dir)

        # 测试窗口滑动
        gt_label_test, gt_box_test, gt_windows_test = video_process(ann_df_test, window_sliding, label_frequency, is_train=False)
        # 保存处理后的测试数据
        with open(os.path.join(info_dir_split, 'window_info_test_{}.log'.format(ca_subject_test)), 'w') as f:
            f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows_test)
        # 处理测试特征
        feature_process(ca_subject_test,
                        gt_label_test, gt_box_test, 
                        gt_windows_test, info_dir_feature, 
                        sample_interval, wind_length, tem_feature_dir,
                        spa_feature_dir, False)


def feature_process(ca_subject_test,
                    gt_label, gt_box, 
                    gt_windows,info_dir_feature, 
                    sample_interval,wind_length,tem_feature_dir,
                    spa_feature_dir, is_train=True):
    """
    根据给定的数据和配置，处理并保存视频特征信息。
    
    参数:
    - ca_subject_test: 测试主体编号，用于路径生成。
    - gt_label: 真实标签列表，用于训练时提供标签信息。
    - gt_box: 真实框列表，用于训练时提供框位置信息。
    - gt_windows: 真实窗口信息，包含开始帧和视频名称。
    - info_dir_feature: 保存处理后特征信息的目录。
    - sample_interval: 采样间隔，用于确定特征的起始和终止索引。
    - wind_length: 窗口长度，用于特征处理的固定长度。
    - tem_feature_dir: 时间特征目录，用于获取时间特征数据。
    - spa_feature_dir: 空间特征目录，用于获取空间特征数据。
    - is_train: 是否为训练模式，决定保存的文件名和处理方式。
    """
    # 根据是否是训练模式，设置子文件夹名称
    if is_train:
        sub_file_name = 'train'
    else:
        sub_file_name = 'test'

    print(sub_file_name, ca_subject_test)
    
    # 生成保存特征的路径，并创建如果不存在
    save_feature_path = os.path.join(info_dir_feature, 'subject_{}'.format(ca_subject_test), sub_file_name)
    if not os.path.exists(save_feature_path):
        os.makedirs(save_feature_path)
    
    # 遍历每个真实窗口信息，处理并保存特征
    for iord, line in enumerate(gt_windows):
        begin_frame, vid_name = line[0], line[1]
        vid_name = vid_name.split('/')[-1]
        last_save_file =os.path.join(save_feature_path, vid_name + '_' + str(begin_frame).zfill(5) + '.npz')
        
        # 根据模式获取标签和信息
        if is_train:
            info = gt_box[iord]
            label = gt_label[iord]
        else:
            info = None
            label = None

        # 计算特征的起始和终止索引
        start_idx = int(int(begin_frame) / sample_interval)
        end_idx = start_idx + wind_length
        
        # 提取视频名称中的子编号
        sub_name = 's' + vid_name.split('_')[0]

        # 处理并获取时间特征
        mode = 'flow'
        feat_file = os.path.join(tem_feature_dir, sub_name, vid_name + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)

        # 处理并获取空间特征
        mode = 'rgb'
        feat_file = os.path.join(spa_feature_dir, sub_name, vid_name + '-' + mode + '.npz')
        feat_spa = construct_feature(vid_name, feat_file, start_idx, end_idx, wind_length)

        # 保存处理后的特征信息到文件
        np.savez(last_save_file,
                vid_name=vid_name,
                begin_frame=int(begin_frame),
                action=info,
                class_label=label,
                feat_tem=feat_tem,
                feat_spa=feat_spa)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='/home/LGSNet/casme2_annotation_357.csv')
    parser.add_argument('--sample_interval', type=int, default=2)
    parser.add_argument('--wind_length', type=int, default=64)
    parser.add_argument('--window_sliding', type=int, default=128)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--label_frequency', type=int, default=1)
    args = parser.parse_args()
    cas_train_test(args.ann_path, 
                    args.sample_interval,
                    args.wind_length,
                    args.window_sliding,
                    args.label_frequency,
                    args.is_train)
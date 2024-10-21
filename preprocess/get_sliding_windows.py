import os
import pickle
import numpy as np
import pandas as pd
import argparse
import random


def window_data(start_frame, ann_df, video_name, ws, label_frequency):
    """
    处理视频帧数据以生成用于模型训练或推理的窗口数据。

    参数:
    - start_frame (int): 当前窗口的起始帧号。
    - ann_df (DataFrame): 包含注释信息的 DataFrame，包括 startFrame, endFrame 和 type_idx 字段。
    - video_name (str): 正在处理的视频名称。
    - ws (int): 窗口大小，即单个窗口包含的帧数。
    - label_frequency (int): 标签频率，用于调整帧范围的尺度。

    返回:
    - label_one_win (ndarray): 每个窗口的标签数组。
    - box_one_win (ndarray): 每个窗口内动作的起始和结束帧数组。
    - window_one_win (list): 包含窗口的起始帧号和视频名称的列表。
    """
    # 计算当前窗口的结束帧号
    window_size = ws
    end_frame = start_frame + window_size

    # 初始化列表以存储每个窗口的动作标签和边界框信息
    label_one_win = list()
    box_one_win = list()
    # 存储窗口的起始帧号和视频名称
    window_one_win = [int(start_frame), video_name]
    # 初始化类别标签，'0' 表示负样本
    class_label = [0, 1, 2]
    # 如果只考虑正样本，使用以下行
    # class_label = [0, 1]

    # 遍历注释 DataFrame 以处理每个动作注释
    for i in range(len(ann_df)):
        # 计算动作的调整后的起始和结束帧
        act_start = ann_df.startFrame.values[i] // label_frequency
        act_end = ann_df.endFrame.values[i] // label_frequency
        # 确保结束帧大于起始帧
        assert act_end > act_start
        # 计算当前窗口与动作的重叠部分
        overlap = min(end_frame, act_end) - max(start_frame, act_start)
        # 计算重叠部分相对于动作持续时间的比例
        overlap_ration = overlap * 1.0 / (act_end - act_start)

        # 定义重叠比例阈值
        overlap_ratio_threshold = 0.9
        # 如果重叠比例超过阈值，认为是相关动作
        if overlap_ration > overlap_ratio_threshold:
            # 计算动作在窗口内的地面真值起始和结束帧
            gt_start = max(start_frame, act_start) - start_frame
            gt_end = min(end_frame, act_end) - start_frame
            # 将动作标签和边界框信息追加到列表中
            label_one_win.append(class_label.index(ann_df.type_idx.values[i]))
            box_one_win.append([gt_start, gt_end])

    # 将边界框信息和标签转换为 ndarray
    box_one_win = np.array(box_one_win).astype('float32')
    label_one_win = np.array(label_one_win)
    # 返回标签、边界框信息和窗口信息
    return label_one_win, box_one_win, window_one_win


def sliding_window(ann_df, video_name, ws, label_frequency, train_sample, is_train, neg):
    window_size = ws
    video_ann_df = ann_df[ann_df.video == video_name]
    frame_count = video_ann_df.frame_num.values[0]//label_frequency
    if is_train:
        stride = int(window_size / (train_sample))
    else:
        stride = int(window_size / (2*train_sample))
  
    num_window = int(1.0*(frame_count + stride - window_size) / stride)
    windows_start = [i * stride for i in range(num_window)]
    if is_train and num_window == 0:
        windows_start = [0]
    if frame_count > window_size:
        windows_start.append(frame_count - window_size)

    label_one_video = list()
    box_one_video = list()
    window_one_video = list()
    for start in windows_start:
        if is_train:
            label_tmp, box_tmp, window_tmp = window_data(start, video_ann_df, video_name, ws, label_frequency)
            if len(label_tmp) > 0:
                label_one_video.append(label_tmp)
                box_one_video.append(box_tmp)
                window_one_video.append(window_tmp)
        else:
            window_one_video.append([int(start), video_name])
    
    # generate more neg samples
    if is_train and neg:
        num_pos = len(label_one_video)
        start_pos = [int(i[0]) for i in window_one_video]
        remain_windows_start = [i for i in windows_start[:-1] if i not in start_pos]
        number_neg = min(len(remain_windows_start), 3*num_pos)
        if remain_windows_start:
            # method 2
            # for i in range(num_pos):
            #     num_tmp = random.randint(0, len(remain_windows_start)-1)
            #     window_one_video.append([int(remain_windows_start[num_tmp]),video_name])
            # method 4-1 & 4-2
            # for i in range(len(remain_windows_start)):
            #     num_tmp = i   
            #     window_one_video.append([int(remain_windows_start[num_tmp]), video_name])
            repeat_list = list()
            for i in range(number_neg):
                if number_neg==len(remain_windows_start):
                    num_tmp = i   
                    window_one_video.append([int(remain_windows_start[num_tmp]), video_name])
                else:
                    num_tmp = random.randint(0, len(remain_windows_start)-1)
                    while num_tmp in repeat_list:
                        num_tmp = random.randint(0, len(remain_windows_start)-1)
                    window_one_video.append([int(remain_windows_start[num_tmp]),video_name])
                    repeat_list.append(num_tmp)
                rnd_tmp = random.randint(0, 100)

                tmp_sss = os.path.basename(os.path.dirname(video_name)) [:4]
                try:
                    tmp_sss = int(tmp_sss)
                    if window_size==128:
                        if  0 < rnd_tmp <= 55:
                            fake_start = random.randint(0, ws-18*ws//128)
                            fake_end= random.randint(fake_start+15*ws//128, min(ws, fake_start+50*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 56 < rnd_tmp <= 82:
                            fake_start = random.randint(0, ws-54*ws//128)
                            fake_end= random.randint(fake_start+50*ws//128, min(ws, fake_start+75*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        else:
                            fake_start = random.randint(0, ws-93*ws//128)
                            fake_end= random.randint(fake_start+75*ws//128, min(ws, fake_start + ws))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                    else:
                        if  0 < rnd_tmp <= 52:
                            fake_start = random.randint(0, ws-18*ws//window_size)
                            fake_end= random.randint(fake_start+15*ws//window_size, min(ws, fake_start+50*ws//window_size))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 53 < rnd_tmp <= 78:
                            fake_start = random.randint(0, ws-54*ws//window_size)
                            fake_end= random.randint(fake_start+50*ws//window_size, min(ws, fake_start+75*ws//window_size))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 79 < rnd_tmp <= 91:
                            fake_start = random.randint(0, ws-93*ws//window_size)
                            fake_end= random.randint(fake_start+75*ws//window_size, min(ws, fake_start+116*ws//window_size))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 92 < rnd_tmp <= 97:
                            fake_start = random.randint(0, ws-155*ws//window_size)
                            fake_end= random.randint(fake_start+116*ws//window_size, min(ws, fake_start+160*ws//window_size))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        else:
                            fake_start = random.randint(0, ws-212*ws//window_size)
                            fake_end= random.randint(fake_start+160*ws//window_size, min(ws, fake_start + ws))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                except:
                    if tmp_sss != 'samm':
                        if  0 < rnd_tmp <= 75:
                            fake_start = random.randint(0, ws-18*ws//128)
                            fake_end= random.randint(fake_start+12*ws//128, min(ws, fake_start+36*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 75 < rnd_tmp <= 94:
                            fake_start = random.randint(0, ws-54*ws//128)
                            fake_end= random.randint(fake_start+36*ws//128, min(ws, fake_start+62*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        else:
                            fake_start = random.randint(0, ws-93*ws//128)
                            fake_end= random.randint(fake_start+62*ws//128, min(ws, fake_start + ws))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                    else:
                        if  0 < rnd_tmp <= 92:
                            fake_start = random.randint(0, ws-18*ws//128)
                            fake_end= random.randint(fake_start+13*ws//128, min(ws, fake_start+23*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        elif 92 < rnd_tmp <= 97:
                            fake_start = random.randint(0, ws-35*ws//128)
                            fake_end= random.randint(fake_start+ 23*ws//128, min(ws, fake_start+64*ws//128))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                        else:
                            fake_start = random.randint(0, ws-96*ws//128)
                            fake_end= random.randint(fake_start+ 64*ws//128, min(ws, fake_start+ws))
                            box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                label_one_win = np.array([0])
                box_one_video.append(box_one_win)
                label_one_video.append(label_one_win)
            
            # for k in range(num_pos): 
                # method 1
                # if ws > 128:
                #     box_one_win = list()
                #     label_one_win = list()
                #     fake_start = random.randint(0, ws-18)
                #     fake_end= random.randint(fake_start+8, ws)
                #     # while (3 <= int(fake_end-fake_start) <= 117) == False:
                #     #     fake_start = random.randint(0, ws-18*ws/128)
                #     #     fake_end= random.randint(fake_start+8*ws/128, ws)
                #     rest = fake_end
                #     box_one_win.append([fake_start, fake_end])
                #     label_one_win.append(0)
                #     while rest + 8 > ws-18:
                #         fake_start = random.randint(rest, ws-18)
                #         fake_end= random.randint(fake_start+8, ws)
                #         rest = fake_end
                #         box_one_win.append([fake_start, fake_end])
                #         label_one_win.append(0)
                #     box_one_win = np.array(box_one_win).astype('float32')
                #     label_one_win = np.array(label_one_win)
                # else:
                
                # method 2: this method is better than others
                # generate the fake length of clips
                # fake_start = random.randint(0, ws-18*ws//128)
                # fake_end= random.randint(fake_start+12*ws//128, min(ws, fake_start+32*ws//128))
                # box_one_win = np.array([[fake_start, fake_end]]).astype('float32')
                # label_one_win = np.array([0])
                # box_one_video.append(box_one_win)
                # label_one_video.append(label_one_win)

                # method 3: this method is worst
                # box_tmp_fake = list()
                # label_tmp_fake = list()
                # num_neg = random_num_times(1, 0.8, 1, 5)
                # start_tmp = list()
                # end_tmp = list()
                # proposal_clips = list()
                # for _ in range(num_neg):
                #     if start_tmp:
                #         for i in range(len(start_tmp)+1):
                #             if i == 0 and start_tmp[i]>=15:
                #                 proposal_clips.append([1,start_tmp[i]])
                #             elif i == len(start_tmp) and end_tmp[i-1]<=128-15:
                #                 proposal_clips.append([end_tmp[i-1],128])
                #             elif 0<i<len(start_tmp) and start_tmp[i] - end_tmp[i-1]>=15 :
                #                 proposal_clips.append([end_tmp[i-1],start_tmp[i]])
                #         if not proposal_clips:
                #             break
                #         else:
                #             random_loc = random.randint(1,len(proposal_clips))
                #             random_clip = proposal_clips[random_loc-1]   
                #             left_tmp = random_clip[0]
                #             left_len = random_clip[1]-9*ws//128
                #             right_len = random_clip[1]
                #     else:
                #         left_tmp = 1
                #         left_len = ws-9*ws//128
                #         right_len = ws
                #     fake_start = random.randint(left_tmp, left_len)
                #     fake_length = random_num_frame(16, 8, 40, 30, 8, 120)
                #     fake_end =  min(right_len, fake_start+fake_length *ws//128)
                #     box_tmp_fake.append([fake_start, fake_end]) 
                #     label_tmp_fake.append(0)  
                #     start_tmp.append(fake_start)
                #     end_tmp.append(fake_end)
                #     start_tmp.sort()
                #     end_tmp.sort() 
                # box_fake = np.array(box_tmp_fake).astype('float32')
                # label_fake = np.array(label_tmp_fake)
                # box_one_video.append(box_fake)
                # label_one_video.append(label_fake)

                # method 4: it created in 5 November 2021
                # according to the distribution of all pos samples, generating negative samples in turn
                # all generated neg samples are remained sliding windows in method 4
                # the number of method 2 is based on number of pos samples

    return label_one_video, box_one_video, window_one_video


def random_num_frame(avg1, stdd1, avg2, stdd2, left, right):
    a = random.sample([random.gauss(avg1, stdd1), random.gauss(avg2, stdd2)],1)[0]
    while (left <= a <= right) == False:
        a = random.sample([random.gauss(avg1, stdd1), random.gauss(avg2, stdd2)],1)[0]
    return int(a)

def random_num_times(avg, stdd, left, right):
    a = random.gauss(avg, stdd)
    while (left <= a <= right) == False:
        a = a = random.gauss(avg, stdd)
    return int(a)


def video_process(ann_df, ws, label_frequency, train_sample=4, is_train=True, neg=True):
    """
    根据视频注释信息处理视频数据，并返回相应的标签、边界框和时间窗口。

    参数:
    - ann_df: DataFrame, 包含视频注释信息的数据框，包括视频名称、标签和边界框坐标等。
    - ws: int, 窗口大小，定义处理视频数据时的时间窗口长度。
    - label_frequency: dict, 包含每个标签频率的字典，用于在训练时平衡数据分布。
    - train_sample: int, 默认 4, 训练时每段视频采样的数量。
    - is_train: bool, 默认 True, 表示当前处理是否为训练数据集。
    - neg: bool, 默认 True, 表示处理时是否包含负样本。

    返回:
    - label: list, 包含所有处理后的视频数据的标签。
    - boxes: list, 包含所有处理后的视频数据的边界框。
    - window: list, 包含所有处理后的视频数据的时间窗口。
    """
    # 去重并确保生成的文件顺序一致
    video_name_list = list(set(ann_df.video.values[:].tolist()))
    video_name_list.sort()

    label = list()
    boxes = list()
    window = list()

    for video_name in video_name_list:
        # 调用 sliding_window 函数处理每个视频，获取临时的标签、边界框和时间窗口
        label_tmp, box_tmp, window_tmp = sliding_window(ann_df, video_name, ws, label_frequency, train_sample, is_train, neg)
        # 如果是训练模式且临时标签列表不为空，则扩展标签、边界框和时间窗口列表
        if is_train and (len(label_tmp) > 0):
            label.extend(label_tmp)
            boxes.extend(box_tmp)
        window.extend(window_tmp)
    return label, boxes, window


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ann_path', type=str, default='./casme2_annotation.csv')
    parser.add_argument('-info_dir', type=str, default='./')
    parser.add_argument('-is_train', type=bool, default=True)
    parser.add_argument('-window_sliding', type=int, default=128)
    parser.add_argument('-frequency', type=int, default=1)
    args = parser.parse_args()

    ann_df = pd.read_csv(args.ann_path)
    gt_label, gt_box, gt_windows = video_process(ann_df, args.window_sliding, args.frequency, is_train=args.is_train)
    if args.is_train:
        with open(os.path.join(args.info_dir, 'gt_label.pkl'), 'wb') as f:
            pickle.dump(gt_label, f)
        with open(os.path.join(args.info_dir, 'gt_box.pkl'), 'wb') as f:
            pickle.dump(gt_box, f)
    with open(os.path.join(args.info_dir, 'window_info.log'), 'w') as f:
        f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows)




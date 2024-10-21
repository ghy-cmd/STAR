import pandas as pd
import os
import argparse


def main(e_path, v_path, l_path):
    """
    主函数，用于处理表情识别数据，生成标注文件。
    
    参数:
    e_path: str, 表情数据的Excel文件路径。
    v_path: str, 视频数据的根目录路径。
    l_path: str, 长度统计的保存路径。
    """
    # 读取Excel文件中的三个sheet
    df_sheet1 = pd.read_excel(e_path, sheet_name=0, header=None)
    df_sheet2 = pd.read_excel(e_path, sheet_name=1, header=None)
    df_sheet3 = pd.read_excel(e_path, sheet_name=2, header=None)

    # 初始化各字段列表
    subject = list()
    video_names = list()
    type_v = list()
    type_idx = list()
    start_frame = list()
    end_frame = list()
    frame_num = list()
    length = list()

    # 遍历第一个sheet，处理每个表情样本
    for i in range(len(df_sheet1[1].values)):
        expression = df_sheet1[1].values[i]
        startframe = df_sheet1[2].values[i]
        endframe = df_sheet1[4].values[i]
        label = df_sheet1[7].values[i]

        # 处理异常的endframe值
        if int(endframe) == 0 or endframe <= df_sheet1[3].values[i]:
            if int(endframe) != 0:
                print('EOF', i, df_sheet1[7].values[i], startframe)
            endframe = df_sheet1[3].values[i] + 20

        # 根据label确定type_idx
        if label == 'macro-expression':
            t_idx = 1
        elif label == 'micro-expression':
            t_idx = 2

        # 确保第一个sheet的第一列和第二个sheet的第二列相等
        assert(int(df_sheet1[0].values[i]) == int(df_sheet2[2].values[i]))
        file_index = df_sheet2[1].values[i]

        # 处理视频路径
        ex = str(expression)
        if ex.split('_')[0] in df_sheet3[1].values:
            num_index = list(df_sheet3[1].values).index(ex.split('_')[0])
            subname = '0' + str(list(df_sheet3[0].values)[num_index])
            subname_full = str(file_index) + '_' + subname
            file_path = os.path.join(v_path, file_index)
            video_name = os.listdir(file_path)
            video_name_part = [name[:7] for name in video_name]

            # 确定视频文件名和完整路径
            if subname_full[1:] in video_name_part:
                file_name = video_name[video_name_part.index(subname_full[1:])]
                path_full = os.path.join(v_path, file_index, file_name) # 通过索引比对确定完整路径
                all_pic = os.listdir(path_full)
                all_pic = [int(str(i).split('.')[0][3:]) for i in all_pic]
                all_pic.sort()
                all_pic = ['/img' + str(i) + '.jpg' for i in all_pic]

                count_pic = len(all_pic)

                # 更新各字段列表
                subject.append(df_sheet1[0].values[i])
                video_names.append(path_full)
                type_v.append(label)
                type_idx.append(t_idx)
                start_frame.append(int(startframe))
                end_frame.append(int(endframe))
                frame_num.append(int(count_pic))
                length.append(int(endframe - startframe))
                print(i)
            else:
                print(i)
                print('11111111111111111', ex[:-2])

    # 创建字典，保存处理后的数据
    dic_inf = dict()
    dic_inf['subject'] = subject
    dic_inf['video'] = video_names
    dic_inf['type'] = type_v
    dic_inf['type_idx'] = type_idx
    dic_inf['startFrame'] = start_frame
    dic_inf['endFrame'] = end_frame
    dic_inf['frame_num'] = frame_num
    dic_inf['length'] = length
    # count_len(length, l_path)

    # 保存为CSV文件
    df = pd.DataFrame(dic_inf, columns=['subject', 'video', 'type', 'type_idx', 'startFrame',  'endFrame', 'frame_num', 'length'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)
    df.to_csv('./casme2_annotation.csv', encoding='utf-8', index=False)

    
def count_len(lenth, length_path):
    """
    统计给定列表中元素的频率，并根据元素的大小分类计数。
    
    此函数首先将给定列表 `lenth` 中的所有元素及其出现次数写入到指定的文件中。
    然后，它计算每个唯一元素出现的次数，并将这些信息写入到 `length_path` 指定的文件中。
    最后，根据元素的大小，将元素分类为 large、middle 和 small，并打印出每类元素的数量。
    
    :param lenth: 包含待统计元素的列表
    :param length_path: 写入统计结果的文件路径
    """
    
    # 将列表 lenth 中的每个元素写入到文件中，每个元素占一行
    with open('./cas(me)2_len_all_1111.txt', 'a') as f:
        for i in lenth:
            f.write(str(i))
            f.write('\n')
    
    # 计算并生成 lenth 列表中每个唯一元素的出现次数的字典
    dicts = {x: lenth.count(x) for x in set(lenth)}
    
    # 将每个唯一元素及其出现次数写入到 length_path 指定的文件中
    with open(length_path, 'a') as f:
        for k, v in dicts.items():
            f.write(str(k))
            f.write(' ')
            f.write(str(v))
            f.write('\n')
    
    # 初始化 large、middle 和 small 的计数器
    large = 0
    middle = 0
    small = 0
    
    # 根据元素的大小，更新相应的计数器
    for i in lenth:
        if i <= 40:
            small = small + 1
        elif 40 < i <= 80:
            middle = middle + 1
        else:
            large = large + 1
    
    # 打印 large、middle 和 small 的数量
    print(large, middle, small)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-excel_path', type=str, default='/home/Data/CAS(ME)^2code_final(Updated).xlsx')
    parser.add_argument('-video_files', type=str, default='/home/Data/rawpic')
    parser.add_argument('-len_path', type=str, default='./cas(me)2_len.txt')
    args = parser.parse_args()
        
    args = parser.parse_args()

    excel_path = args.excel_path
    video_files = args.video_files
    len_path = args.len_path

    main(excel_path, video_files, len_path)

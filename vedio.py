import cv2
import os

# 图像帧文件夹路径
frame_folder = '/home/Data/rawpic/s15/15_0101disgustingteeth'

# 输出视频文件路径
output_video_path = '/home/Data/output_video.mp4'

# 获取文件夹中的所有图像文件路径
frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 按文件名排序（假设文件名是按顺序命名的）
frame_files.sort()

# 读取第一帧以确定视频的宽度和高度
first_frame = cv2.imread(frame_files[0])
height, width, layers = first_frame.shape

# 设置视频的帧率
fps = 30

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 遍历每个图像文件并写入视频
for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video_writer.write(frame)

# 释放视频写入对象
video_writer.release()

print(f"Video saved to {output_video_path}")
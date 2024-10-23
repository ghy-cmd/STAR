import cv2
import os

# 图像帧文件夹路径
folder_path = '/home/Data/test/s15/15_0102eatingworms'

# 输出视频文件路径
output_video_path = '/home/Data/output_video_landmark_align.mp4'

# 获取文件夹中的所有图像文件路径
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files= [int(str(i).split('.')[0][3:]) for i in image_files]
image_files.sort()
image_files = ['img' + str(i).zfill(3) + '.jpg' for i in image_files]
image_files = [os.path.join(folder_path, i) for i in image_files]

# 读取第一帧以确定视频的宽度和高度
first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape

# 设置视频的帧率
fps = 30

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 遍历每个图像文件并写入视频
for frame_file in image_files:
    frame = cv2.imread(frame_file)
    video_writer.write(frame)

# 释放视频写入对象
video_writer.release()

print(f"Video saved to {output_video_path}")
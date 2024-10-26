from pathlib import Path
from typing import Union

import cv2


class FrameSampler:

    def __init__(self, video_file: Union[str, Path], sampling_fps: int | float = None):
        """
        采样器
        :param video_file: 视频文件
        :param sampling_fps:
        """
        self.video_stream = video_stream = cv2.VideoCapture(str(video_file))
        # 源视频参数
        self.total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps = video_stream.get(cv2.CAP_PROP_FPS)  # 帧率: float
        # 采样参数
        if sampling_fps is not None:
            sampling_fps = min(fps, max(1., sampling_fps))
            self.interval = max(1, int(fps / sampling_fps))
        else:
            self.interval = 1
        # 迭代器索引：第一帧为0
        self.frame_idx = 0
        # 视频分辨率
        self.frame_w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def resolution(self):
        """返回视频分辨率: H,W"""
        return self.frame_h, self.frame_w

    def __enter__(self):
        """with语句将触发"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        frm = self.next()
        if frm is None:
            raise StopIteration
        return frm

    def __len__(self):
        # 返回总行数，供tqdm使用
        return self.total_frames

    def next(self):
        if not self.video_stream.isOpened():
            return None

        while True:
            ret, frm = self.video_stream.read()
            if ret:
                curr_idx = self.frame_idx
                self.frame_idx += 1  # 从0开始
                if curr_idx % self.interval == 0:
                    return frm
            else:
                break
        self.video_stream.release()  # ret=False时提前关闭流,避免忘记调用self.close()
        return None

    def all(self):
        frames = []
        while True:
            frm = self.next()
            if frm is not None:
                frames.append(frm)
            else:
                break
        return frames

    def close(self):
        if self.video_stream:
            self.video_stream.release()
            self.video_stream = None
        self.frame_idx = 0
        self.interval = 1


def merge_videos_with_sliding_line(src_video, dst_video, output_path, thickness=3, speedx=1.0, offset=0):
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(src_video)
    cap2 = cv2.VideoCapture(dst_video)

    # 获取视频的宽度、高度和帧率
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # 确保两个视频的分辨率和帧率一致
    if (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)) != width or
            int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height or
            cap2.get(cv2.CAP_PROP_FPS) != fps):
        print("两个视频的分辨率或帧率不一致，无法合成。")
        cap1.release()
        cap2.release()
        return

    # 获取视频的总帧数
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 计算竖线每帧的移动距离
    speed = width / (total_frames - offset)
    speed *= speedx

    for frame_index in range(total_frames):
        # 读取两个视频的当前帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果任何一个视频到达末尾，停止处理
        if not ret1 or not ret2:
            break

        # 创建一个新的帧用于合成
        blended_frame = frame1.copy()
        if frame_index >= offset:
            # 更新竖线位置
            line_pos = int((frame_index - offset) * speed)

            # 将视频2的画面拷贝到竖线左侧
            blended_frame[:, :line_pos] = frame2[:, :line_pos]

            # 在合成帧上绘制白色竖线
            cv2.line(blended_frame, (line_pos, 0), (line_pos, height), (255, 255, 255), thickness)

        # 写入输出视频
        out.write(blended_frame)

        # 显示合成后的帧（可选）
        # cv2.imshow('Blended Video', blended_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放所有资源
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

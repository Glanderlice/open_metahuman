import cv2


class FrameSampler:

    def __init__(self, video_file: str, sampling_fps: int | float = 1):
        self.video_stream = video_stream = cv2.VideoCapture(video_file)
        # 源视频参数
        self.total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps = video_stream.get(cv2.CAP_PROP_FPS)  # 帧率: float
        # 采样参数
        sampling_fps = min(fps, max(1., sampling_fps))
        self.interval = max(1, int(fps / sampling_fps))
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

import os
import pickle
import time
from collections import deque
from concurrent.futures import as_completed
from ctypes import cdll
from pathlib import Path
from typing import List, Tuple, Union, Any

import cv2
import gradio as gr
import numpy as np

from modules.face_module.face_analyze import FaceAnalyzer, Face
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.embedders import ArcFaceOnnx
from modules.face_module.face_model.enhancers import GFPGANOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.face_module.face_model.swappers import INSwapperOnnx
from modules.face_module.face_recog import FaceRecognizer, SimpleFaceDB, FaceMeta
from tool.file_util import make_dir, file_hash, delete_dir
from tool.multi_task import MultiThread
from tool.video_helper import FrameSampler

# opencv导出.mp4视频时依赖的外部DLL动态库
app_root = Path(__file__).parent.parent

cache_root = app_root / '.cache'
example_root = app_root / 'examples'

# dll_path = app_root / 'resources/openh264-1.8.0-win64.dll'
# if os.path.exists(dll_path):
#     cdll.LoadLibrary(str(dll_path))


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Engine:

    def __init__(self):
        self.model_root = str(app_root / 'models/face_models')
        self.model_index = {}

        self.processor = None
        self.recognizer = None

    def get_swapper(self, device='cuda'):
        swapper = self.model_index.get("swapper", None)
        if swapper is None:
            swapper = INSwapperOnnx(f"{self.model_root}/inswapper_128.onnx", device)
            self.model_index["swapper"] = swapper
        return swapper

    def get_enhancer(self, device='cuda'):
        enhancer = self.model_index.get("enhancer", None)
        if enhancer is None:
            enhancer = GFPGANOnnx(f"{self.model_root}/GFPGANv1.4.onnx", 100, device)
            self.model_index["enhancer"] = enhancer
        return enhancer

    def get_processor(self, device='cuda'):
        if self.processor is None:
            detector = self.model_index.get("detector", None)
            if detector is None:
                detector = RetinaFaceOnnx(f"{self.model_root}/det_10g.onnx", device, (640, 640), 5)  # 人脸检测
                self.model_index["detector"] = detector

            landmarker = self.model_index.get("landmarker", None)
            if landmarker is None:
                landmarker = Landmark3d68ONNX(f"{self.model_root}/1k3d68.onnx", device)  # 关键点标记
                self.model_index["landmarker"] = landmarker

            embedder = self.model_index.get("embedder", None)
            if embedder is None:
                embedder = ArcFaceOnnx(f"{self.model_root}/w600k_r50.onnx", device)  # 人脸向量化
                self.model_index["embedder"] = embedder

            self.processor = FaceAnalyzer().add_model(detector, "detector").add_model(landmarker,
                                                                                      "landmarker").add_model(embedder,
                                                                                                              "embedder")

        return self.processor

    def get_recognizer(self, device='cpu'):
        if self.recognizer is None:
            embedder = self.model_index.get("embedder", None)
            if embedder is None:
                embedder = ArcFaceOnnx(f"{self.model_root}/w600k_r50.onnx", device)  # 人脸向量化
                self.model_index["embedder"] = embedder
            self.recognizer = FaceRecognizer(embedder, SimpleFaceDB(update_scheme='pose'))
        return self.recognizer

    def release(self, model_names: Union[str, List[str], None] = None) -> None:
        if isinstance(model_names, str) and model_names:
            model_names = [model_names]

        deletes = model_names if model_names is not None else list(self.model_index.keys())

        for model_name in deletes:
            if model_name in self.model_index:
                del self.model_index[model_name]
                print(f"released model: {model_name}")

        self.processor = None
        self.recognizer = None

        return None


def blend_curve(i, start_value, stop_value, start_frame, stop_frame):
    if i < start_frame:
        return start_value
    elif i < stop_frame:
        return start_value + (i - start_frame) * (stop_value - start_value) / float(stop_frame - start_frame)
    else:
        return stop_value


def scan(video_file: Union[str, Path], sampling_fps: Union[int, float] = 2, k: int = 10, use_cache: bool = True,
         progress=gr.Progress()):
    if video_file is None:
        return []
    # 单例模式调用, 节省资源
    progress(0, desc="准备模型")

    processor = Engine().get_processor('cuda')
    recognizer = Engine().get_recognizer()

    video_id = file_hash(video_file)  # 使用视频内容Hash命名视频临时文件夹
    make_dir(str(cache_root / video_id), not use_cache)

    db_path = str(cache_root / video_id / f'faces_fps{sampling_fps}.pkl')
    if not recognizer.facedb.load_db(db_path):  # data_root / 'faces.pkl'
        with FrameSampler(video_file=video_file, sampling_fps=sampling_fps) as sampler:
            for i, frame in enumerate(sampler):
                faces = processor.apply(frame)  # 面部检测-标记-向量
                for face in faces:
                    ret: int = recognizer.learn(face)  # 会返回添加结果
                progress((sampler.frame_idx, sampler.total_frames), desc="扫描人脸")
            # 序列化识别结果
            recognizer.facedb.save_db(db_path)

    # 人脸出现频率排序
    most_faces = recognizer.most_frequent_faces(k=k)
    most_faces = [(data.image[:, :, ::-1], data.pid) for data, _ in most_faces]  # BGR->RGB

    return most_faces


def extract(video_file: Union[str, Path], progress=gr.Progress()):
    """把视频中所有帧的人脸数据以[(0,List[Face]), (1,List[Face]), ...]保存到文件"""
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    frame_dir = str(video_dir / 'frames')
    make_dir(frame_dir, clear=False)

    progress(0, desc="准备模型")
    processor = Engine().get_processor()

    file_names = []

    with FrameSampler(video_file=video_file) as sampler:
        buffer = []
        for i, frame in enumerate(progress.tqdm(sampler, desc="分析每一帧中人脸")):
            faces = processor.apply(frame, ['detector', 'embedder'])  # 跳过关键点扫描
            if faces:
                buffer.append((i, faces))
                if len(buffer) == 1200:
                    file_name = f"frames_{i:06}.pkl"
                    with open(f"{frame_dir}/{file_name}", 'wb') as file:
                        pickle.dump(buffer, file)
                        file_names.append(file_name)
                    buffer.clear()
        if len(buffer) > 0:
            file_name = f"{i:06}.pkl"
            with open(f"{frame_dir}/{file_name}", 'wb') as file:
                pickle.dump(buffer, file)
                file_names.append(file_name)
    return file_names


def recognize(video_file: Union[str, Path], src_person, progress=gr.Progress()):
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    frame_dir = str(video_dir / 'frames')

    progress(0, desc="准备模型")
    recognizer: FaceRecognizer = Engine().get_recognizer()

    file_names = sorted(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else None
    if not file_names:
        file_names = extract(video_file)

    buffer = []
    for file_name in file_names:
        file_path = f"{frame_dir}/{file_name}"
        with open(file_path, 'rb') as file:
            frames_data = pickle.load(file)
        # 第i帧包含的所有faces
        for (i, faces) in progress.tqdm(frames_data, desc=f"识别人物-{file_name}"):  # (int, List[Face])
            target_faces = []
            for face in faces:
                # 如果命中目标person则保留帧序号和对应face信息
                person: str = recognizer.search(face, i, [src_person])  # src_person缩小追踪范围
                if person == src_person:
                    target_faces.append((face, person))
            if target_faces:
                buffer.append((i, target_faces))
    return buffer  # (i:int, List[face:Face, person:str])


def _binary_search(lst: List[Tuple[int, Any]], x) -> int:
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid][0] < x:
            low = mid + 1
        elif lst[mid][0] > x:
            high = mid - 1
        else:
            return mid  # 找到了
    return -1  # 未找到


class FrameCacheSys:

    def __init__(self, frame_dir: str, max_len: int = 2):
        self.file_index = []
        file_names = sorted(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else []
        if file_names:
            self.file_index = [[int(file_name.split(".")[0]), f"{frame_dir}/{file_name}", None] for file_name in
                               file_names]

        self.queue = deque(maxlen=max_len)
        self.max_len = max_len

    def _enqueue(self, item) -> int:
        pop_item = self.queue.popleft() if len(self.queue) == self.max_len else None
        self.queue.append(item)  # 插入新元素到队尾
        return pop_item  # 返回被弹出的元素, 如果有

    def get_data(self, frame_id: int):
        if frame_id is not None and frame_id >= 0:
            for fi, data in enumerate(self.file_index):
                d, path, buffer = data
                if frame_id <= d:
                    if not buffer:
                        with open(path, 'rb') as file:
                            buffer = pickle.load(file)
                            data[2] = buffer
                        if buffer:
                            pop_frame = self._enqueue(fi)
                            if pop_frame is not None:
                                self.file_index[pop_frame][2] = None
                    p = _binary_search(buffer, frame_id)
                    j, data = buffer[p]
                    assert j == frame_id
                    return data  # (i, faces)
        return None


def swap(video_file: Union[str, Path], src_person, dst_image, enhance: bool = True, swap_progress=True,
         thread_num=4, buffer_size=16, progress=gr.Progress()):
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    swap_dir = str(video_dir / 'swap')
    make_dir(swap_dir, clear=True)

    filter_faces = recognize(video_file, src_person)

    progress(0, desc="准备模型")
    Engine().release()
    swapper = Engine().get_swapper()

    progress(0, desc="向量化上传图片")
    to_face: Face = image2face(dst_image)

    threads = thread_num  # 线程数：2,4,6,8
    batch_size = buffer_size  # 任务等待队列大小: 16, 24, 32, 48
    interval = 0.1  # 休眠间隔:单位秒

    executor = MultiThread(threads)

    with FrameSampler(video_file=video_file) as sampler:

        if sampler.total_frames < 5 * sampler.fps:
            swap_progress = False

        futures = []

        n: int = 0
        for i, frame in enumerate(sampler):
            j, faces = filter_faces[n]
            if i < j:
                continue
            elif i == j:
                # 处理未完成的任务
                while len(futures) >= batch_size:
                    completed_futures = as_completed(futures)
                    for completed_future in completed_futures:
                        futures.remove(completed_future)  # 移除已完成的任务
                    if len(futures) <= threads:  # 确保不会多次移除
                        break
                    time.sleep(interval)

                blend_value = blend_curve(i, 0.1, 1.0, int(0.5 * sampler.fps),
                                          int(3 * sampler.fps)) if swap_progress else 1.0
                image_path = f"{swap_dir}/{i:06}.png"

                # swap_func(frame, faces, to_face, blend_value, image_path)

                future = executor.submit(swapper.swap_frame, frame, faces, to_face, blend_value, image_path)
                futures.append(future)

                n += 1
                progress((n, len(filter_faces)), desc="换脸")
                if n == len(filter_faces):
                    break
            else:
                raise ValueError('i > j is impossible')
        # 等待所有剩余任务完成
        for future in futures:
            try:
                future.result()  # 获取剩余任务的结果
            except Exception as e:
                print(f"任务出现错误: {e}")

    Engine().release()

    image_list = sorted(os.listdir(swap_dir)) if os.path.isdir(swap_dir) else []

    if not image_list:
        return None

    enhancer = Engine().get_enhancer() if enhance else None

    output_path = video_dir / f"output.mp4"
    with FrameSampler(video_file=video_file) as sampler:
        n: int = 0
        progress(0, desc="开始合成")
        export_stream = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), sampler.fps,
                                        (sampler.frame_w, sampler.frame_h))
        futures = []
        if enhance:
            frame_cache = FrameCacheSys(str(video_dir / 'frames'), max_len=2)
        for i, frame in enumerate(progress.tqdm(sampler, desc="人脸超分" if enhance else "合成中")):
            if n < len(image_list):
                image_name = image_list[n]
                j = int(image_name.split(".")[0])
            if i == j:
                frame = cv2.imread(f"{swap_dir}/{image_name}")
                n += 1
            if enhance:
                faces = frame_cache.get_data(i)
                if not faces:
                    export_stream.write(frame)
                    continue

                future = executor.submit(enhancer.enhance_frame, frame, faces)
                futures.append(future)
                if len(futures) >= batch_size:
                    for future in futures:
                        try:
                            frame = future.result()  # 获取剩余任务的结果
                            export_stream.write(frame)
                        except Exception as e:
                            print(f"人脸超分任务出现错误: {e}")
                    futures.clear()
            else:
                export_stream.write(frame)
        # 完成剩余的任务
        if futures:
            for future in futures:
                try:
                    frame = future.result()  # 获取剩余任务的结果
                    export_stream.write(frame)
                except Exception as e:
                    print(f"人脸超分任务出现错误: {e}")
        export_stream.release()

    Engine().release()

    return output_path


def select_src_face(select_evt: gr.SelectData):
    person_id = select_evt.value.get('caption', None)
    data: FaceMeta = Engine().get_recognizer().facedb.get(person_id)
    return data.image[:, :, ::-1], person_id


def image2face(image: np.ndarray):
    """输入的gr.Image是channel=RGB的ndarray"""
    if isinstance(image, str) and os.path.isfile(image):
        image = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        image = image[:, :, ::-1]  # RGB->BGR
    processor = Engine().get_processor()
    face = processor.apply(image)[0]
    return face


def clear_cache():
    delete_dir(cache_root, False)
    return gr.update(value=None)


def unload():
    Engine().release()


def web_app():
    make_dir(cache_root, clear=False)

    with gr.Blocks() as face_app:
        # panel1
        video_in = gr.Video(label="输入视频", height=640, autoplay=True)
        # panel2
        with gr.Accordion("点击展开/收起扫描参数", open=True):
            with gr.Row(equal_height=True):
                video_in_examples = gr.Examples(label="系统示例",
                                                examples=[
                                                    f"{str(example_root / 'video')}/swap.mp4",
                                                    f"{str(example_root / 'video')}/girls.mp4",
                                                    f"{str(example_root / 'video')}/dance.mp4"
                                                ],
                                                inputs=video_in,
                                                )
                with gr.Column():
                    max_scans = gr.Slider(label="人脸上限", minimum=1, maximum=8, step=1, value=4)
                    sample_fps = gr.Slider(label="采样帧率", minimum=1, maximum=3, step=1, value=2)
        # panel3
        gallery = gr.Gallery(label="发现的人脸", height=200, columns=8, allow_preview=False, interactive=False)

        # panel4
        with gr.Accordion("点击展开/收起换脸参数", open=True):
            with gr.Row(equal_height=True):
                with gr.Column():
                    src_face = gr.Image(label="选择视频人物", height=240, interactive=False)
                    src_name = gr.Textbox(label="视频人物ID", interactive=False)
                with gr.Column():
                    dst_face = gr.Image(label="换脸图片", height=240)
                    dst_face_examples = gr.Examples(label="系统示例",
                                                    examples=[
                                                        f"{str(example_root / 'image')}/jay.png",
                                                        f"{str(example_root / 'image')}/lyf.png",
                                                        f"{str(example_root / 'image')}/wyz.png"
                                                    ],
                                                    inputs=dst_face,
                                                    )

                with gr.Column(scale=3):
                    with gr.Row():
                        face_sr = gr.Checkbox(label="人脸超分", value=True)
                        swap_prog = gr.Checkbox(label="渐变换脸", value=False)
                    thread_num = gr.Slider(label="多线程个数", minimum=2, maximum=8, step=1, value=6)
                    buffer_size = gr.Slider(label="缓存队列长", minimum=16, maximum=32, step=2, value=24)
                    swap_button = gr.Button("开始换脸")

        # panel5
        video_out = gr.Video(label="输出视频", height=640, autoplay=True)

        # panel6
        with gr.Row():
            clear_button = gr.Button("清理缓存")
            release_button = gr.Button("卸载模型")

        # listener监听事件：
        video_in.change(scan, [video_in, sample_fps, max_scans], [gallery])
        gallery.select(select_src_face, None, [src_face, src_name])
        swap_button.click(swap, [video_in, src_name, dst_face, face_sr, swap_prog, thread_num, buffer_size], video_out)
        clear_button.click(clear_cache, None, video_in)
        release_button.click(unload)

    face_app.launch(allowed_paths=[str(cache_root), str(example_root)])


if __name__ == '__main__':
    web_app()

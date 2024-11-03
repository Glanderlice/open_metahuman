import os
import pickle
from ctypes import cdll
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

from modules.face_module.face_analyze import FaceAnalyzer, Face, draw_on
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.embedders import ArcFaceOnnx
from modules.face_module.face_model.enhancers import GFPGANOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.face_module.face_model.swappers import INSwapperOnnx
from modules.face_module.face_recog import FaceRecognizer, SimpleFaceDB, FaceMeta
from tool.file_util import clear_dir, make_dir, file_hash, delete_dir
from tool.video_helper import FrameSampler, merge_videos_with_sliding_line

import gradio as gr

# opencv导出.mp4视频时依赖的外部DLL动态库
app_root = Path(__file__).parent.parent

cache_root = app_root / '.cache'

dll_path = app_root / 'resources/openh264-1.8.0-win64.dll'
assert os.path.exists(dll_path), '找不到动态库openh264-1.8.0-win64.dll'
if os.path.exists(dll_path):
    cdll.LoadLibrary(str(dll_path))  # 如果 DLL 文件存在，加载它
    print(f"DLL 文件 {dll_path} 已加载")
else:
    print(f"DLL 文件 {dll_path} 不存在")


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

    def get_processor(self, device='cpu'):
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

            self.processor = FaceAnalyzer().add_model(detector).add_model(landmarker).add_model(embedder)

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
        deletes = model_names if model_names is not None else [self.model_index.keys()]
        for model_name in deletes:
            if model_name in self.model_index:
                del self.model_index[model_name]
                print(f"released model: {model_name}")
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
    # 单例模式调用, 节省资源
    processor = Engine().get_processor()
    recognizer = Engine().get_recognizer()

    video_id = file_hash(video_file)  # 使用视频内容Hash命名视频临时文件夹
    make_dir(str(cache_root / video_id), not use_cache)

    db_path = str(cache_root / video_id / 'faces.pkl')
    if not recognizer.facedb.load_db(db_path):  # data_root / 'faces.pkl'
        progress(0, desc="开始扫描")
        with FrameSampler(video_file=video_file, sampling_fps=sampling_fps) as sampler:
            for i, frame in enumerate(progress.tqdm(sampler)):
                faces = processor.apply(frame)  # 面部检测-标记-向量
                for face in faces:
                    ret: int = recognizer.learn(face)  # 会返回添加结果
            # 序列化识别结果
            recognizer.facedb.save_db(db_path)
    else:
        progress(100, desc="加载存档")

    # 人脸出现频率排序
    most_faces = recognizer.most_frequent_faces(k=k)
    most_faces = [(data.image[:, :, ::-1], data.pid) for data, _ in most_faces]  # BGR->RGB

    return most_faces


def extract(video_file: Union[str, Path], progress=gr.Progress()):
    """把视频中所有帧的人脸数据以[(0,List[Face]), (1,List[Face]), ...]保存到文件"""
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    frame_dir = str(video_dir / 'frames')

    processor = Engine().get_processor()

    file_names = []

    with FrameSampler(video_file=video_file) as sampler:
        progress(0, desc="开始分析")
        buffer = []
        for i, frame in enumerate(progress.tqdm(sampler, desc="分析每一帧")):
            faces = processor.apply(frame)
            if faces:
                buffer.append((i, faces))
                if len(buffer) == 2400:
                    with open(f"{frame_dir}/frames_{i:06}.pkl", 'wb') as file:
                        pickle.dump(buffer, file)
                    buffer.clear()
        if len(buffer) > 0:
            file_name = f"frames_{i:06}.pkl"
            with open(f"{frame_dir}/{file_name}", 'wb') as file:
                pickle.dump(buffer, file)
                file_names.append(file_name)

    return file_names


def recognize(video_file: Union[str, Path], src_person, progress=gr.Progress()):
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    frame_dir = str(video_dir / 'frames')

    recognizer: FaceRecognizer = Engine().get_recognizer()

    file_names = os.listdir(frame_dir).sort()
    if not file_names:
        file_names = extract(video_file)

    buffer = []
    for file_name in file_names:
        file_path = f"{frame_dir}/{file_name}"
        with open(file_path, 'rb') as file:
            frames_data = pickle.load(file)
        # 第i帧包含的所有faces
        for i, faces in progress(frames_data, desc="识别每帧人脸"):  # (int, List[Face])
            for face in faces:
                # 如果命中目标person则保留帧序号和对应face信息
                person: str = recognizer.search(face)
                if person == src_person:
                    buffer.append((i, face, person))
    return buffer  # (i:int, face:Face, person:str)


def _first_consecutive_frames(frames):
    start, last, length = -1, -1, 0
    if frames:
        for i, _, _ in frames:
            if last + 1 < i:
                break
            start = i if start == -1 else start
            last = i
            length += 1
    return start, length


def swap2(video_file: Union[str, Path], src_person, dst_image, swap_progress=True, progress=gr.Progress()):
    video_id = file_hash(video_file)
    video_dir = cache_root / video_id
    frame_dir = str(video_dir / 'frames')

    swapper: INSwapperOnnx = Engine().get_swapper()
    to_face: Face = image2face(dst_image)

    src_faces = recognize(video_file, src_person)

    with FrameSampler(video_file=video_file) as sampler:

        if sampler.total_frames < 5 * sampler.fps:
            swap_progress = False

        for i, frame in enumerate(progress.tqdm(sampler, desc="处理中")):
            faces = processor.apply(frame)

            blend_value = blend_curve(i, 0.1, 1.0, int(0.67 * sampler.fps),
                                      int(3 * sampler.fps)) if swap_progress else 1.0

            for face in faces:
                # 照片换脸
                person: str = recognizer.search(face)
                if person == src_person:
                    frame = swapper.apply_one(face, frame, to_face,
                                              blend=blend_value)  # blend = blend_curve(i, 0.1, 1.0, 20, 90)

    # 使用 PNG 格式保存，无损压缩
    output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def swap(video_file: Union[str, Path], src_person, dst_image, enhance: bool = True, swap_progress: bool = True,
         use_cache: bool = True, progress=gr.Progress()):
    processor: FaceAnalyzer = Engine().get_processor()
    recognizer: FaceRecognizer = Engine().get_recognizer()

    swapper: INSwapperOnnx = Engine().get_swapper()

    enhancer = Engine().get_enhancer() if enhance else Engine().release("enhancer")  # 不用就释放enhancer

    to_face: Face = image2face(dst_image)

    video_id = file_hash(video_file)  # 使用视频内容Hash命名视频临时文件夹
    # make_dir(str(cache_root / video_id), not use_cache)
    output_path = cache_root / video_id / f"output.mp4"

    with FrameSampler(video_file=video_file) as sampler:
        progress(0, desc="开始处理")
        video_codec = cv2.VideoWriter_fourcc(*'avc1')
        export_stream = cv2.VideoWriter(output_path, video_codec, sampler.fps,
                                        (sampler.frame_w, sampler.frame_h))

        if sampler.total_frames < 5 * sampler.fps:
            swap_progress = False

        for i, frame in enumerate(progress.tqdm(sampler, desc="处理中")):
            faces = processor.apply(frame)

            blend_value = blend_curve(i, 0.1, 1.0, int(0.67 * sampler.fps),
                                      int(3 * sampler.fps)) if swap_progress else 1.0

            for face in faces:
                # 照片换脸
                person: str = recognizer.search(face)
                if person == src_person:
                    frame = swapper.apply_one(face, frame, to_face,
                                              blend=blend_value)  # blend = blend_curve(i, 0.1, 1.0, 20, 90)
                # 人脸超分
                if enhance:
                    frame = enhancer.enhance(face, frame)

            export_stream.write(frame)
        export_stream.release()

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


def unload():
    Engine().release()


def web_app():
    make_dir(cache_root, clear=False)

    with gr.Blocks() as face_app:
        # panel1
        video_in = gr.Video(label="输入视频", height=640, autoplay=True)
        # panel2
        with gr.Accordion("点击展开扫描参数", open=False):
            with gr.Row():
                max_scans = gr.Slider(
                    label="人脸上限",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=4,
                )
                sample_fps = gr.Slider(
                    label="采样帧率",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=2,
                )
        # panel3
        gallery = gr.Gallery(label="发现的人脸", height=200, columns=8, allow_preview=False, interactive=False)

        # panel4
        with gr.Accordion("点击展开换脸参数", open=False):
            with gr.Row(equal_height=True):
                with gr.Column():
                    src_face = gr.Image(label="视频人物图片", interactive=False)
                    src_name = gr.Textbox(label="视频人物ID", interactive=False)
                dst_face = gr.Image(label="换脸图片")
                with gr.Column():
                    face_sr = gr.Checkbox(label="人脸超分", value=True)
                    swap_prog = gr.Checkbox(label="开启渐变", value=True)
            swap_button = gr.Button("开始换脸")

        # panel5
        video_out = gr.Video(label="输出视频", height=640, autoplay=True)

        # panel6
        with gr.Row():
            clear_button = gr.Button("清理缓存")
            release_button = gr.Button("卸载模型")

        # listener监听事件：
        video_in.upload(scan, [video_in, sample_fps, max_scans], [gallery])
        gallery.select(select_src_face, None, [src_face, src_name])
        swap_button.click(swap, [video_in, src_name, dst_face, face_sr, swap_prog], video_out)
        clear_button.click(clear_cache)
        release_button.click(unload)

    face_app.launch(allowed_paths=[str(cache_root)])


if __name__ == '__main__':
    web_app()

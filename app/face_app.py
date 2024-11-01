import os
from ctypes import cdll
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from typer import Option

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


def start_engine(model_root: Union[str, Path], facedb_path: Union[str, Path] = None):
    # 1.人脸检测模型
    # 1) 人脸检测
    detector = RetinaFaceOnnx(model_file=model_root / 'det_10g.onnx', device='cpu', det_size=(640, 640),
                              max_faces=5)  # 人脸检测
    # 2) 关键点标记
    landmarker = Landmark3d68ONNX(model_file=model_root / '1k3d68.onnx', device='cuda')  # 关键点标记
    # 3) 向量化
    embedder = ArcFaceOnnx(model_file=model_root / 'w600k_r50.onnx', device='cuda')  # 人脸向量化

    processor = FaceAnalyzer()
    processor.add_model(detector, 'detector')
    processor.add_model(landmarker, 'landmarker')
    processor.add_model(embedder, 'embedder')

    # 2.人脸识别模型
    # 依赖：1) 人脸库  2) 向量化模型
    facedb = SimpleFaceDB(update_scheme='pose')  # update_scheme='pose'或'last'
    if facedb_path and os.path.isfile(facedb_path):
        facedb.load_db(facedb_path)
    recognizer = FaceRecognizer(embedder, facedb)

    engine = Engine()
    engine.processor = processor
    engine.recognizer = recognizer
    return engine


def scan_faces(video_file: Union[str, Path], engine: Engine, sampling_fps: Union[int, float] = 2, k: int = 10):
    processor = engine.processor
    recognizer: FaceRecognizer = engine.recognizer
    with FrameSampler(video_file=video_file, sampling_fps=sampling_fps) as sampler:
        for i, frame in enumerate(sampler):
            faces = processor.apply(frame)  # 面部检测-标记-向量
            for face in faces:
                ret: int = recognizer.learn(face)  #
    # 人脸出现频率排序
    most_faces: List[Tuple[str, int]] = recognizer.most_frequent_faces(k=k)

    # *某些场景可能需要以下接口：
    # 导出所有人脸图片
    # recognizer.facedb.export_face_images(data_root / 'faces')
    # 序列化/反序列化识别结果
    # recognizer.facedb.save_db(data_root / 'faces.pkl')
    # recognizer.facedb.load_db(data_root / 'faces.pkl')

    faces = []
    for person, _ in most_faces:
        face = recognizer.facedb.get(person)
        if face is not None:
            faces.append((face, person))

    return faces


def image2face(image_file: Union[str, Path], engine: Engine):
    image = cv2.imread(str(image_file))
    processor = engine.processor
    face = processor.apply(image)[0]
    return face


def blend_curve(i, start_value, stop_value, start_frame, stop_frame):
    if i < start_frame:
        return start_value
    elif i < stop_frame:
        return start_value + (i - start_frame) * (stop_value - start_value) / float(stop_frame - start_frame)
    else:
        return stop_value


def postprocess_faces(video_file: Union[str, Path], engine: Engine, swap_faces: Dict[str, Face], enhance: bool = True,
                      save_as: Union[str, Path] = None, extra: Dict[str, Any] = None):
    processor: FaceAnalyzer = engine.processor
    recognizer: FaceRecognizer = engine.recognizer
    swapper: INSwapperOnnx = engine.swapper if swap_faces else None
    enhancer: GFPGANOnnx = engine.enhancer if enhance else None

    extra = extra if extra is not None else {}

    export_stream = None

    with FrameSampler(video_file=video_file) as sampler:
        if save_as is not None:
            save_as = str(save_as)
            export_stream = cv2.VideoWriter(save_as, cv2.VideoWriter_fourcc(*'avc1'), sampler.fps,
                                            (sampler.frame_w, sampler.frame_h))
        # parse extra parameters
        draw_rect = extra.get('draw_rect', False)
        save_frames_dir = str(extra.get('save_frame', None))
        if save_frames_dir:
            clear_dir(save_frames_dir)

        for i, frame in enumerate(tqdm(sampler, desc="Processing frames")):
            faces = processor.apply(frame)
            for face in faces:
                # 照片换脸
                if swapper:
                    person: str = recognizer.search(face)
                    to_face: Face = swap_faces.get(person, None)
                    if to_face is not None:
                        frame = swapper.apply_one(face, frame, to_face,
                                                  blend=1.0)  # blend = blend_curve(i, 0.1, 1.0, 20, 90)
                # 人脸超分
                if enhancer:
                    frame = enhancer.enhance(face, frame)

            if draw_rect:
                frame = draw_on(frame, faces)
            if save_frames_dir:
                cv2.imwrite(f"{save_frames_dir}/frame_{i:06}.jpg", frame)

            if export_stream is not None:
                export_stream.write(frame)

    if export_stream is not None:
        export_stream.release()


def rename_faces(engine: Engine, names: List[Tuple[str, str]]):
    facedb: SimpleFaceDB = engine.recognizer.facedb
    for old_name, new_name in names:
        if not facedb.rename(old_name, new_name):
            print(f"renaming {old_name} to {new_name} failed.")


def save_facedb(engine: Engine, db_path: Union[str, Path]):
    """保存人脸库"""
    facedb: SimpleFaceDB = engine.recognizer.facedb
    facedb.save_db(db_path=db_path)


def merge_videos(video_src: Union[str, Path], video_dst: Union[str, Path], video_output: Union[str, Path]):
    merge_videos_with_sliding_line(video_src, video_dst, video_output, speedx=3, offset=0)


def run():
    model_root = app_root / 'models/face_models'
    data_root = app_root / 'data'

    # 启动人脸检测/识别模型和人脸库
    engine = start_engine(model_root, facedb_path=data_root / 'db/facedb.pkl')

    # 遍历视频记录人脸
    faces: List[Tuple[str, int]] = scan_faces(data_root / 'video/movie.mp4', engine=engine, sampling_fps=2, k=10)
    rename_faces(engine, [('face_0', 'gold_woman'), ('face_1', 'natasha')])

    # 人脸置换
    engine.swapper = INSwapperOnnx(model_root / 'inswapper_128.onnx', device='gpu')
    dst_face = image2face(data_root / 'image/jay.png', engine=engine)

    # 人脸超清
    engine.enhancer = GFPGANOnnx(model_root / 'GFPGANv1.4.onnx', device='gpu')

    # 后处理：人脸置换, 人脸超分
    swap_scheme = {'gold_woman': dst_face, 'natasha': dst_face}
    options = {'save_frame': data_root / 'temp/frames', 'draw_rect': False}

    save_path = data_root / 'temp/output.mp4'
    postprocess_faces(data_root / 'video/movie.mp4', engine, swap_faces=swap_scheme, enhance=False, save_as=save_path,
                      extra=options)

    # merge_videos()

    print('end')


def scan(video_file: Union[str, Path], sampling_fps: Union[int, float] = 2, k: int = 10, use_cache: bool = True,
         progress=gr.Progress()):
    # 单例模式调用, 节省资源
    processor = Engine().get_processor()
    recognizer = Engine().get_recognizer()

    # video_id = os.path.basename(video_file)
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


def swap(video_file: Union[str, Path], src_person, dst_image, enhance: bool = True, swap_progress: bool = True,
         use_cache: bool = True, progress=gr.Progress()):
    processor: FaceAnalyzer = Engine().get_processor()
    recognizer: FaceRecognizer = Engine().get_recognizer()

    swapper: INSwapperOnnx = Engine().get_swapper()

    enhancer = Engine().get_enhancer() if enhance else Engine().release("enhancer")  # 不用就释放enhancer

    to_face: Face = image_to_face(dst_image)

    video_id = file_hash(video_file)  # 使用视频内容Hash命名视频临时文件夹
    # make_dir(str(cache_root / video_id), not use_cache)
    output_path = cache_root / video_id / f"output.mp4"

    with FrameSampler(video_file=video_file) as sampler:
        progress(0, desc="开始处理")
        video_codec = cv2.VideoWriter_fourcc(*"mp4v")  # cv2.VideoWriter_fourcc(*'avc1')
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


def image_to_face(image: np.ndarray):
    """输入的gr.Image是channel=RGB的ndarray"""
    if isinstance(image, str) and os.path.isfile(image):
        image = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        image = image[:, :, ::-1]  # RGB->BGR
    processor = Engine().get_processor()
    face = processor.apply(image)[0]
    return face


def web_app():
    make_dir(cache_root, clear=False)
    # delete_dir(cache_root)

    with gr.Blocks() as face_app:
        state = gr.State(value={})
        video_in = gr.Video(label="输入视频", height=640, autoplay=True)
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
        gallery = gr.Gallery(label="扫描到的人脸", height=200, columns=8, allow_preview=False, interactive=False)
        # with gr.Column():
        #     output_video = gr.Video(label="", streaming=True, autoplay=True)

        video_in.upload(
            fn=scan,
            inputs=[video_in, sample_fps, max_scans],
            outputs=[gallery]
        )

        with gr.Accordion("点击展开换脸参数", open=False):
            with gr.Row(equal_height=True):
                with gr.Column():
                    src_face = gr.Image(label="视频人物图片", interactive=False)
                    src_name = gr.Textbox(label="视频人物ID", interactive=False)
                dst_face = gr.Image(label="换脸图片")
                with gr.Column():
                    face_sr = gr.Checkbox(label="人脸超分", value=True)
                    swap_prog = gr.Checkbox(label="渐变开启", value=True)
            swap_button = gr.Button("开始换脸")

        video_out = gr.Video(label="输出视频", height=640, autoplay=True)

        # listener监听事件：
        gallery.select(select_src_face, None, [src_face, src_name])
        swap_button.click(swap, [video_in, src_name, dst_face, face_sr, swap_prog], video_out)

    face_app.launch()


if __name__ == '__main__':
    web_app()

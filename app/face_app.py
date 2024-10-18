import os
from ctypes import cdll
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import cv2
from tqdm import tqdm

from modules.face_module.face_analyze import FaceProcessor, Face, draw_on
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.embedders import ArcFaceOnnx
from modules.face_module.face_model.enhancers import GFPGANOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.face_module.face_model.swappers import INSwapperOnnx
from modules.face_module.face_recog import FaceRecognizer, SimpleFaceDB
from utils.file_util import clear_dir
from utils.video_helper import FrameSampler, merge_videos_with_sliding_line

# opencv导出.mp4视频时依赖的外部DLL动态库
app_root = Path(__file__).parent.parent
dll_path = app_root / 'resources/openh264-1.8.0-win64.dll'
assert os.path.exists(dll_path), '找不到动态库openh264-1.8.0-win64.dll'
if os.path.exists(dll_path):
    cdll.LoadLibrary(str(dll_path))  # 如果 DLL 文件存在，加载它
    print(f"DLL 文件 {dll_path} 已加载")
else:
    print(f"DLL 文件 {dll_path} 不存在")


class Engine:
    def __init__(self):
        self.processor = None
        self.recognizer = None
        self.swapper = None
        self.enhancer = None


def start_engine(model_root: Union[str, Path], facedb_path: Union[str, Path] = None):
    # 1.人脸检测模型
    # 1) 人脸检测
    detector = RetinaFaceOnnx(model_file=model_root / 'det_10g.onnx', device='cpu', det_size=(640, 640),
                              max_faces=5)  # 人脸检测
    # 2) 关键点标记
    landmarker = Landmark3d68ONNX(model_file=model_root / '1k3d68.onnx', device='cpu')  # 关键点标记
    # 3) 向量化
    embedder = ArcFaceOnnx(model_file=model_root / 'w600k_r50.onnx', device='cpu')  # 人脸向量化

    processor = FaceProcessor()
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
    recognizer = engine.recognizer
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

    return most_faces


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
    processor: FaceProcessor = engine.processor
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


if __name__ == '__main__':
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

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from modules.face_module.face_analyze import FaceProcessor, Face
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.embedders import ArcFaceOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.face_module.face_recog import FaceRecognizer, SimpleFaceDB
from utils.image_helper import reverse_channel
from utils.video_helper import FrameSampler


def face_scan_image(img, max_faces=10, channel_order='BGR', detector: FaceProcessor = None):
    """
    收集图片中所有的人脸
    :return:
    """
    # 点击时先清空人脸识别器的缓存, 删除记录的人脸信息
    global recognizer

    recognizer.clear()

    if isinstance(img, str) and os.path.isfile(img):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray) and channel_order == 'RGB':
        img = reverse_channel(img)  # RGB->BGR
    else:
        raise TypeError('img must be str or ndarray in BGR/RGB format')

    faces: List[Face] = detector.apply(img)
    recognizer.remember_faces(faces, position=None)

    if image_data is not None:
        if isinstance(image_data, str) and os.path.isfile(image_data):
            image_data = cv2.imread(image_data)
        if isinstance(image_data, np.ndarray):
            image_data = reverse_channel(image_data)  # RGB->BGR
            faces = frame_swapper.analyzer.apply(image_data, max_faces=keep_faces_num, vector=False)
            frame_swapper.recognizer.remember_faces(faces, position=None)
    elif video_file:
        with FrameExtractor(video_file, sample_fps=sampling_fps) as extractor:  # 抽帧
            total_frames = extractor.total_frames
            fps = extractor.fps

            max_frames = total_frames - fps
            min_frames = min(int(5 * fps), max_frames)
            end_frame = int(sampling_range * 0.01 * total_frames)
            if end_frame > max_frames:
                end_frame = None
            elif end_frame < min_frames:
                end_frame = min_frames

            for frame in extractor:  # 遍历所有帧
                frame_idx = extractor.frame_idx
                if end_frame and frame_idx > end_frame:
                    break
                time_in_sec = int(extractor.frame_idx / fps)  # 当前帧位于第几秒(extractor.frame_idx从0开始)
                faces = frame_swapper.analyzer.apply(frame, max_faces=keep_faces_num, vector=False)
                frame_swapper.recognizer.remember_faces(faces, position=time_in_sec)
                if time_in_sec % int(5 * sampling_fps) == 0:
                    print(f'scanning faces in frame[{frame_idx}] at {time_in_sec} seconds of video')

    faces = frame_swapper.recognizer.facedb.most_frequent_faces(k=keep_faces_num)  # 返回找到的脸,最多5个
    persons, face_images = [], []
    for face, freq in faces:
        persons.append(face.person)
        gender = ''
        if face.gender is not None:
            gender = '男' if face.gender else '女'
        age = '?'
        if face.age is not None:
            age = face.age
        face_images.append((face.image[:, :, ::-1], f'{face.person}:{gender} {age}岁 {freq}次'))

    return gr.Gallery.update(value=face_images)


def click_face_scan_btn(image_data, video_file, sampling_fps, sampling_range, keep_faces_num):
    pass


if __name__ == '__main__':
    app_root = Path(__file__).parent.parent
    model_root = app_root / 'models/face_detect'
    detector = RetinaFaceOnnx(model_file=model_root / 'det_10g.onnx', device='cpu', det_size=(640, 640),
                              max_faces=5)  # 人脸检测
    landmarker = Landmark3d68ONNX(model_file=model_root / '1k3d68.onnx', device='cpu')  # 关键点标记
    embedder = ArcFaceOnnx(model_file=model_root / 'w600k_r50.onnx', device='cpu')  # 人脸向量化

    engine = FaceProcessor()
    engine.add_model(detector)
    engine.add_model(landmarker)
    engine.add_model(embedder)

    # 遍历一个视频
    data_root = app_root / 'data'

    facedb = SimpleFaceDB(update_scheme='pose')  # update_scheme='pose'或'last'
    recognizer = FaceRecognizer(embedder, facedb)

    # 记录人脸
    with FrameSampler(video_file=str(data_root / 'video' / 'movie.mp4'), sampling_fps=2) as sampler:
        for i, frame in enumerate(sampler):
            # cv2.imwrite(f'frame{i}.jpg', frame)
            faces = engine.apply(frame)  # 面部检测-标记-向量
            for face in faces:
                ret: int = recognizer.learn(face)
            if i % 5 == 0:
                facedb.export_face_images(data_root / 'faces')

    most_faces: List[Tuple[str, int]] = recognizer.most_frequent_faces()
    recognizer.facedb.save_db(data_root / 'faces.pkl')

    recognizer.facedb.export_face_images(data_root / 'faces')

    recognizer.facedb.load_db(data_root / 'faces.pkl')


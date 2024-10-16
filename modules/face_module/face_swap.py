import os.path
import threading
from typing import List

import numpy as np

from core.common.gpu_uitl import check_gpu
from core.common.multi_task import MultiThread
from core.face_module.face_analysis import FaceRecognizer, FaceAnalyzer
from core.face_module.face_meta import Face
from core.face_module.image_enhance import FaceEnhancer
from deepfacelab.dfm_swap import DFMSwapper
from insightface.onnx_model.inswapper import INSwapper
from mediana.image_util import diag2rect
from mediana.timer import timing

lock = threading.Lock()


class FaceSwapper:

    def __init__(self, model_dir, dfm_model_dir=None):
        assert model_dir and os.path.isdir(model_dir)
        self.model_dir = model_dir
        analyzer_model_dir = model_dir
        self.analyzer = FaceAnalyzer(analyzer_model_dir, landmark=True, attribute=True, vector=True)

        arcface_model_path = os.path.join(model_dir, 'glintr100.onnx')
        self.recognizer = FaceRecognizer(arcface_model_path)

        assert model_dir is None or os.path.isdir(model_dir)



        inswap_model_path = os.path.join(model_dir, 'inswapper_128.onnx')
        self.pic_swapper = INSwapper(inswap_model_path)

        self.face_enhancer = None
        self.enhance_target = False

        self.max_threads = None

        # self.gpu_info = {}
        self._record_gpu_usage()

    def reload(self):
        """使用记录的地址重新加载所有模型, 并且记录GPU信息"""
        with lock:
            if self.analyzer:
                self.analyzer.reload()
            if self.recognizer:
                self.recognizer.facedb.reload()
            if self.dfm_swapper:
                self.dfm_swapper.reload()
            if self.pic_swapper:
                self.pic_swapper.reload()
            if self.face_enhancer:
                self.face_enhancer.reload()
        # 记录GPU信息：此时只会记录模型占用量
        return self._record_gpu_usage()

    def multi_threads(self, max_workers):
        self.max_threads = max_workers
        MultiThread(self.max_threads)
        return self

    def load_face_enhancer(self, model_path=None, blend=100, target_only=False):
        with lock:
            self.enhance_target = target_only  # 仅针对目标人脸(指换脸的那些)
            if self.face_enhancer is not None:
                if self.face_enhancer.blend != blend:
                    self.face_enhancer.blend = blend
                return self.face_enhancer
            if model_path is None:
                model_path = os.path.join(self.model_dir, 'GFPGANv1.4.onnx')
            assert model_path and os.path.isfile(model_path)
            self.face_enhancer = FaceEnhancer(model_path, blend=blend)
        return self.face_enhancer

    def release_face_enhancer(self):
        if self.face_enhancer:
            self.face_enhancer = None

    def get_face_vec(self, img):
        """使用analyzer中的arcface模型的输出作为向量返回, 如果要用于inswap换脸模型的输入必须使用w600k_r50.onnx这个模型"""
        face = self.analyzer.apply(img, 1, False, False, True)
        if face:
            face_vec = face[0].vec
        else:
            face_vec = None
        return face_vec

    @timing(interval=1)
    def apply2image(self, src_img: np.ndarray, swap_scheme: dict, max_detect=1):
        merge_img = src_img

        faces = self.analyzer.apply(src_img, max_detect, False, False, False)

        for i, face in enumerate(faces):
            person = self.recognizer.recognize(face.bbox, face.picture)
            is_target = 0
            if person in swap_scheme:
                algo, parameter = swap_scheme.get(person)
                if algo is not None and parameter is not None:
                    merge_img = self.pic_swapper.swap(merge_img, face.kps, parameter, True)
            if self.face_enhancer:
                if not self.enhance_target or is_target > 0:
                    enhancer: FaceEnhancer = self.face_enhancer
                    merge_img = enhancer.enhance(face, merge_img)

        # 面部追踪衰减：如果不衰减则会
        self.recognizer.countdown_tracker()

        return merge_img



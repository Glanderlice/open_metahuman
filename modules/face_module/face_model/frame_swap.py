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


class FrameFaceSwapper:

    def __init__(self, model_dir, dfm_model_dir=None):
        assert model_dir and os.path.isdir(model_dir)
        self.model_dir = model_dir
        analyzer_model_dir = model_dir
        self.analyzer = FaceAnalyzer(analyzer_model_dir, landmark=True, attribute=True, vector=True)

        arcface_model_path = os.path.join(model_dir, 'glintr100.onnx')
        self.recognizer = FaceRecognizer(arcface_model_path)

        assert model_dir is None or os.path.isdir(model_dir)

        self.dfm_swapper = DFMSwapper(model_dir, dfm_model_dir,
                                      max_model_parallel=5) if dfm_model_dir is not None else None

        inswap_model_path = os.path.join(model_dir, 'inswapper_128.onnx')
        self.pic_swapper = INSwapper(inswap_model_path)

        self.face_enhancer = None
        self.enhance_target = False

        self.max_threads = None

        # self.gpu_info = {}
        self._record_gpu_usage()

    def _record_gpu_usage(self):
        """只会在模型重载的时候更新gpu_info"""
        usage, used_gb, total_gb, free_gb = check_gpu()
        self.gpu_info = used_gb
        return usage, used_gb, total_gb, free_gb

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
                    if algo == 'dfm':
                        merge_img, temp_data = self.dfm_swapper.swap(merge_img, diag2rect(face.bbox), parameter, False,
                                                                     'g')
                        is_target = 1
                    elif algo == 'pic':
                        merge_img = self.pic_swapper.swap(merge_img, face.kps, parameter, True)
                        is_target = 2
                    else:
                        raise ValueError(f'invalid model tag: {algo}')
            if self.face_enhancer:
                if not self.enhance_target or is_target > 0:
                    enhancer: FaceEnhancer = self.face_enhancer
                    merge_img = enhancer.enhance(face, merge_img)

        # 面部追踪衰减：如果不衰减则会
        self.recognizer.countdown_tracker()

        return merge_img

    # @timing(interval=10)
    def _divide_tasks(self, faces: List[Face], swap_scheme: dict):
        dfm_rec, dfm_mod = [], []
        pic_kps, pic_vec = [], []
        face_tag = []  # 0=未命中, 1=dfm, 2=pic

        for i, face in enumerate(faces):
            face_tag.append(0)
            person = self.recognizer.recognize(face.bbox, face.picture)
            if person not in swap_scheme:
                continue
            algo, parameter = swap_scheme.get(person)
            if algo is None or parameter is None:
                continue
            if algo == 'dfm':
                dfm_rec.append(diag2rect(face.bbox))  # 先bbox->矩形框, 再添加到list
                dfm_mod.append(parameter)
                face_tag[i] = 1
            elif algo == 'pic':
                pic_kps.append(face.kps)
                pic_vec.append(parameter)
                face_tag[i] = 2
            else:
                raise ValueError(f'invalid model tag: {algo}')

        self.recognizer.countdown_tracker()  # 面部追踪衰减：如果不衰减则会

        return dfm_rec, dfm_mod, pic_kps, pic_vec, face_tag

    @timing(interval=5)
    def _swap_parallel(self, executor, data_buffer):
        futures = []
        for data in data_buffer:
            img, _, pic_kps, pic_vec, _ = data
            future = executor.submit(self.pic_swapper.swap_many, img, pic_kps, pic_vec) if pic_kps else None
            futures.append(future)
        for i, future in enumerate(futures):
            if future:
                output = future.result()  # .result()会阻塞等待该future的执行结果
                data_buffer[i][0] = output
        futures.clear()
        return data_buffer

    @timing(interval=5)
    def _enhance_parallel(self, executor, data_buffer):
        futures = []
        for data in data_buffer:
            img, faces, _, _, tags = data
            tags = tags if self.enhance_target else None
            future = executor.submit(self.face_enhancer.enhance_faces, faces, img, tags) if faces else None
            futures.append(future)
        for i, future in enumerate(futures):
            if future:
                output = future.result()  # .result()会阻塞等待该future的执行结果
                data_buffer[i][0] = output
        futures.clear()
        return data_buffer

    @timing(interval=5)
    def _apply_dfm(self, images, swap_scheme, max_detect):
        data_buffer = []

        for src_img in images:  # 逐帧处理
            faces = self.analyzer.apply(src_img, max_detect, False, False, False)  # 单帧人脸检测：检测框+关键点
            dfm_rec, dfm_mod, pic_kps, pic_vec, face_tag = self._divide_tasks(faces, swap_scheme)  # 单帧任务分离

            merge_img = self.dfm_swapper.swap_many(src_img, dfm_rec, dfm_mod)
            data_buffer.append([merge_img, faces, pic_kps, pic_vec, face_tag])

        return data_buffer

    @timing(interval=5)
    def apply2frames(self, src_img_list: List[np.ndarray], swap_scheme: dict, max_detect=1):
        """
        批处理：一次处理一批图片的人脸识别-变脸-超清
        :param src_img_list: 输入图片集
        :param swap_scheme: 变脸策略, 包括dfm, pic两种
        :param max_detect: 单图最大检测人脸数
        :return:
        """
        data_buffer = self._apply_dfm(src_img_list, swap_scheme, max_detect)

        executor = MultiThread()  # max_workers需要在别处设置

        data_buffer = self._swap_parallel(executor, data_buffer)

        if self.face_enhancer:
            data_buffer = self._enhance_parallel(executor, data_buffer)

        merge_img_list = [data[0] for data in data_buffer]

        data_buffer.clear()

        return merge_img_list

    def auto_release(self, reserve_rate=0.1):
        is_released = False
        last_used_gb = self.gpu_info
        if last_used_gb > 0:
            usage, used_gb, total_gb, free_gb = check_gpu()
            d_used_gb = used_gb - last_used_gb  # 当前比上一次记录点新增消耗的GPU
            gb_thread = round(d_used_gb / self.max_threads, 2) if self.max_threads else 0
            print(f'每个线程消耗GPU：{gb_thread} GB/thread')
            future_left_gb = free_gb - d_used_gb  # 预计下一次计算后剩余的显存
            if future_left_gb < total_gb * reserve_rate:  # 预计下一次是否能够资源(即运行后剩余量不低于GPU的5%)
                self.reload()  # 重载模型, 这将直接释放之前的session所占用的GPU显存
                is_released = True
        return is_released

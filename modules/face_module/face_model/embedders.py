import os
from typing import List, Dict, Any

import cv2
import numpy as np

from modules.face_module import face_align
from modules.face_module.face_analyze import FaceModel, Face
from tool.onnx_helper import build_session
from tool.timer import timing


class ArcFaceOnnx(FaceModel):
    def __init__(self, model_file=None, device='cuda'):
        assert model_file and os.path.exists(model_file)  # glintr100.onnx对侧脸识别比默认的w600k_r50.onnx识别更准确,但照片换脸依赖w600k_r50.onnx
        self.model_file = model_file
        self.device = device
        print(f'arcface model: {model_file}')

        self.input_mean = 127.5
        self.input_std = 127.5

        self.session = None
        self.session = build_session(model_file, device)

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def reload(self):
        self.unload()
        self.session = build_session(self.model_file, self.device)

    def unload(self):
        if self.session:
            self.session = None

    # @timing()
    def apply(self, faces: List[Face] = None, src_img: np.ndarray = None, extra: Dict[str, Any] = None) -> List[Face]:
        """大约10ms(video1.mp4)"""
        if faces:
            for face in faces:
                aligned_face = face.face_img
                if aligned_face is None:  # 如果没有aligned人脸图片,则重新根据kps生成
                    aligned_face = self.crop_align(src_img, face.kps)  # 裁剪->校正
                    face.face_img = aligned_face
                # aligned人脸图片->向量化
                vec = self.vectorize(aligned_face)
                face.vec = vec.flatten()
        return faces

    def crop_align(self, img, kps):
        """
        裁剪校正得到人脸区域
        :param img: 原图像帧
        :param kps: retinaface的5个关键点
        :return: 矫正后的112x112的人脸区域
        """
        return face_align.norm_crop(img, landmark=kps, image_size=self.input_size[0])  # 从原图像帧中裁剪和矫正人脸

    def vectorize(self, align_face) -> np.ndarray:
        """
        向量化
        :param align_face: 使用face_align.norm_crop裁剪校正后的脸, ndarray或list(ndarray)
        :return: 向量, shape=(n,512), n为输入脸的个数
        """
        if not isinstance(align_face, list):
            align_face = [align_face]
        resize = None
        for img in align_face:
            h, w = img.shape[:2]
            if (w, h) != self.input_size:
                resize = self.input_size
                break
        blob = cv2.dnn.blobFromImages(align_face, 1.0 / self.input_std, resize,
                                      (self.input_mean, self.input_mean, self.input_mean),
                                      swapRB=True)  # blob.shape=(1,3,112,112)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out  # net_out.shape=(N,512), N为align_faces数

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

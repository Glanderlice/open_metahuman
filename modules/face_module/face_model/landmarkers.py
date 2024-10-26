import os
import pickle
from typing import List, Dict, Any

import cv2
import numpy as np
import onnx

__all__ = [
    'Landmark3d68ONNX',
]

from modules.face_module import face_align
from modules.face_module.face_analyze import FaceModel, Face
from tool.onnx_helper import build_session


class Landmark3d68ONNX(FaceModel):
    def __init__(self, model_file=None, device='cuda'):
        assert model_file and os.path.exists(model_file)  # 如果未找到模型文件,则尝试在同级目录下再寻找一次det_10g.onnx
        self.model_file = model_file
        self.device = device
        print(f'landmark model: {model_file}')

        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid < 3 and node.name == 'bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', model_file, self.input_mean, self.input_std)
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
        output_shape = outputs[0].shape
        self.require_pose = False
        # print('init output_shape:', output_shape)
        if output_shape[1] == 3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            self.mean_lmk = get_object(os.path.join(os.path.dirname(self.model_file), 'meanshape_68.pkl'))  # 特殊模型依赖
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1] // self.lmk_dim
        self.task = 'landmark_%dd_%d' % (self.lmk_dim, self.lmk_num)

    def reload(self):
        self.unload()
        self.session = build_session(self.model_file, self.device)

    def unload(self):
        if self.session:
            self.session = None

    def apply(self, faces: List[Face] = None, src_img: np.ndarray = None, extra: Dict[str, Any] = None) -> List[Face]:
        if faces:
            for face in faces:
                face.landmark, face.pose = self.mark(src_img, face.bbox)  # pose只有在3D关键点标记才有值,否则返回None
        return faces

    def mark(self, img, bbox):
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)

        pose = None
        if self.require_pose:
            P = face_align.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = face_align.P2sRt(P)
            rx, ry, rz = face_align.matrix2angle(R)
            pose = np.array([rx, ry, rz], dtype=np.float32)  # pitch, yaw, roll
        return pred, pose


def get_object(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np


class Face:

    def __init__(self):
        # 人脸图片(例如采用arcface裁剪矫正后的人脸尺寸为112x112)
        self.face_img = None

        # 人脸检测框(采用的格式：对角线[x1,y1,x2,y2], 即[左上,右下])
        self.bbox = None  # 可扩展项：retinaface检测时会同时输出5个关键点, 使用add_attribute 添加 self.kps
        self.kps = None

        # 人脸关键点(不同模型会输出68,192,468个关键点)
        self.landmark = None  # 可扩展项：face3d68检测时会同时输出头部姿态, 使用add_attribute 添加 self.pose

        # 人脸向量
        self.vec = None

        # 人脸姿态
        self.pose = None

    def add_attribute(self, key: str, value: Any):
        """动态添加新属性, 如:\n
        *基础关键点kps(retinaface检测时会同时输出5个关键点)
        *头部姿态pose(landmark模型的额外输出)\n
        *性别年龄age,gender等等"""
        setattr(self, key, value)

    def deep_copy(self):
        """返回Face对象的深拷贝"""
        return deepcopy(self)


class FaceModel(ABC):
    """Processor抽象类，定义了reload、unload和apply抽象方法"""

    @abstractmethod
    def reload(self):
        """重载模型的方法"""
        pass

    @abstractmethod
    def unload(self):
        """卸载模型的方法"""
        pass

    @abstractmethod
    def apply(self, faces: List[Face], src_img: np.ndarray = None, extra: Dict[str, Any] = None) -> List[Face]:
        """接收一个Face对象并返回一个处理后的Face对象，默认在输入Face上直接修改"""
        pass


class FaceProcessor:
    """
    人脸分析器：对单个图片进行人脸检测, 并对每个识别出的人脸进行关键点标记、性别年龄估计、向量化, 并输出List[Face]
    可以直接init初始化所有模型, 模型是默认的.onnx, 除检测模型是初始化阶段必须的外, 其他的模型可以不实例化, 而是自定义进行单独模型的加载
    """

    def __init__(self):
        """
        人脸检测(使用retinaface含5个简单关键点)-标记关键点/性别年龄分析-人脸向量
        """
        self.models = []
        self.model_index = {}

    def add_model(self, model: FaceModel, model_name: Optional[str] = None):
        self.models.append(model)
        if model_name:
            assert model_name not in self.model_index
            self.model_index[model_name] = model
        return self

    def reload(self):
        """释放->重新加载所有模型"""
        if self.models:
            for model in self.models:
                model.reload()
        return self

    def unload(self):
        """释放所有模型, 但保留引用"""
        if self.models:
            for model in self.models:
                model.unload()
        return self

    # @timing(interval=5)
    def apply(self, img, model_names: List[str] = None) -> List[Face]:
        """
        应用模型
        :param img:
        :param model_names:
        :return:
        """
        faces = []
        models = self.models
        if model_names:
            models = []
            for model_name in model_names:
                assert model_name in self.model_index
                models.append(self.model_index[model_name])
        if models:
            extra_info = {}
            for processor in self.models:
                faces = processor.apply(faces, img, extra_info)

        return faces


def draw_on(img, faces):
    import cv2
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(int)
            # print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                           2)
        # if face.gender is not None and face.age is not None:  # 输出年龄和性别
        #     cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
        #                 0.7, (0, 255, 0), 1)
    return dimg

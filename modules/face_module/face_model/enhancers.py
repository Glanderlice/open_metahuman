import os.path
from typing import Tuple, List

import cv2
import numpy as np

from modules.face_module.face_analyze import Face
from utils.onnx_helper import build_session


def prepare_crop_frame(crop_frame: np.ndarray) -> np.ndarray:
    crop_frame = crop_frame[:, :, ::-1] / 255.0  # BGR先转RGB, 再归一化到[0,1]
    crop_frame = (crop_frame - 0.5) / 0.5  # 归一化到[-1,1]
    crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis=0).astype(np.float32)  # 将channel的维度放到前面,适配torch
    return crop_frame


def normalize_crop_frame(crop_frame: np.ndarray) -> np.ndarray:
    crop_frame = np.clip(crop_frame, -1, 1)
    crop_frame = (crop_frame + 1) / 2
    # crop_frame = crop_frame.transpose(1, 2, 0)
    crop_frame = (crop_frame * 255.0).round()
    crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
    return crop_frame


def paste_back(temp_frame: np.ndarray, crop_frame: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
    crop_frame_height, crop_frame_width = crop_frame.shape[0:2]
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
    inverse_mask = np.ones((crop_frame_height, crop_frame_width, 3), dtype=np.float32)
    inverse_mask_frame = cv2.warpAffine(inverse_mask, inverse_affine_matrix, (temp_frame_width, temp_frame_height))
    inverse_mask_frame = cv2.erode(inverse_mask_frame, np.ones((2, 2)))
    inverse_mask_border = inverse_mask_frame * inverse_crop_frame
    inverse_mask_area = np.sum(inverse_mask_frame) // 3
    inverse_mask_edge = int(inverse_mask_area ** 0.5) // 20
    inverse_mask_radius = inverse_mask_edge * 2
    inverse_mask_center = cv2.erode(inverse_mask_frame, np.ones((inverse_mask_radius, inverse_mask_radius)))
    inverse_mask_blur_size = inverse_mask_edge * 2 + 1
    inverse_mask_blur_area = cv2.GaussianBlur(inverse_mask_center, (inverse_mask_blur_size, inverse_mask_blur_size), 0)
    temp_frame = inverse_mask_blur_area * inverse_mask_border + (1 - inverse_mask_blur_area) * temp_frame
    temp_frame = temp_frame.clip(0, 255).astype(np.uint8)
    return temp_frame


def blend_frame(temp_frame: np.ndarray, paste_frame: np.ndarray, face_enhancer_blend: int) -> np.ndarray:
    face_enhancer_blend = 1 - (face_enhancer_blend / 100)
    temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
    return temp_frame


class GFPGANOnnx:

    def __init__(self, model_file=None, blend=100, device='cuda'):
        assert model_file and os.path.exists(model_file)
        self.model_file = model_file
        self.device = device
        print(f'face enhance model: {model_file}')

        self.blend = blend

        self.session = build_session(model_file, device)

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.output_names = output_names
        self.output_shape = outputs[0].shape

    def reload(self):
        self.session = build_session(self.model_file, self.device)

    def unload(self):
        if self.session:
            self.session = None

    def warp_face(self, target_face: Face, temp_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        template = np.array(
            [
                [192.98138, 239.94708],
                [318.90277, 240.1936],
                [256.63416, 314.01935],
                [201.26117, 371.41043],
                [313.08905, 371.15118]
            ])
        affine_matrix = cv2.estimateAffinePartial2D(target_face.kps, template, method=cv2.LMEDS)[0]
        crop_frame = cv2.warpAffine(temp_frame, affine_matrix, self.input_shape[2:])  # (512, 512)
        return crop_frame, affine_matrix

    def enhance(self, target_face: Face, input_frame: np.ndarray) -> np.ndarray:
        crop_frame, affine_matrix = self.warp_face(target_face, input_frame)
        crop_frame = prepare_crop_frame(crop_frame)

        pred = self.session.run(self.output_names, {self.input_name: crop_frame})[0]
        crop_frame = pred.transpose((0, 2, 3, 1))[0]
        crop_frame = normalize_crop_frame(crop_frame)  # 裁剪的面部区域
        paste_frame = paste_back(input_frame, crop_frame, affine_matrix)  # 面部区域粘贴回原图
        input_frame = blend_frame(input_frame, paste_frame, self.blend)
        return input_frame

    # @timing(interval=10, gpu=True)
    def enhance_faces(self, target_faces: List[Face], input_frame: np.ndarray, mask=None):
        # 无法改成batch推理：Name:'/final_linear/Gemm' Status Message: GEMM: Dimension mismatch, W: {8192,4096} K: 24576 N:8192
        if mask:
            assert len(mask) == len(target_faces)
        out_frame = input_frame
        for i, target_face in enumerate(target_faces):
            if not mask or mask[i] > 0:
                out_frame = self.enhance(target_face, out_frame)
        return out_frame


if __name__ == '__main__':
    src_image = cv2.imread('D:/PycharmProjects/aurora-studio/data/image/1.jpg')

    analyzer_model_dir = 'D:/PycharmProjects/aurora-studio/models/face/weights'
    analyzer = GFPGANOnnx(analyzer_model_dir, landmark=True, attribute=True, vector=True)

    faces = analyzer.apply(src_image, max_faces=10, vector=False)

    enhancer = GFPGANOnnx('D:/PycharmProjects/aurora-studio/models/face/weights/GFPGANv1.4.onnx')
    out_image = src_image
    out_image = enhancer.enhance_faces(faces, out_image)
    # for face in faces:
    #     out_image = enhancer.enhance(face, out_image)
    cv2.imwrite('D:/PycharmProjects/aurora-studio/data/image/enhanced.jpg', out_image)
    print('end.')

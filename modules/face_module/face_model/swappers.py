import os
from typing import List, Dict, Any

import cv2
import numpy as np
import onnx

from onnx import numpy_helper

from modules.face_module import face_align
from modules.face_module.face_analyze import FaceModel, Face
from utils.onnx_helper import build_session


class INSwapperOnnx(FaceModel):
    def __init__(self, model_file=None, device='cuda'):
        assert model_file and os.path.exists(model_file)  # 如果未找到模型文件,则尝试在同级目录下再寻找一次det_10g.onnx
        self.model_file = model_file
        self.device = device
        print(f'inswapper model: {model_file}')

        model = onnx.load(model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])

        self.input_mean = 0.0
        self.input_std = 255.0

        self.session = build_session(self.model_file, self.device)

        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names

        assert len(self.output_names) == 1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        # print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])

    def reload(self):
        self.session = build_session(self.model_file, self.device)

    def unload(self):
        if self.session:
            self.session = None

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
        return pred

    def apply(self, faces: List[Face] = None, src_img: np.ndarray = None, to_face: Face = None, extra: Dict[str, Any] = None):
        merged_img = src_img
        for face in faces:
            merged_img = self.swap(merged_img, face.kps, to_face.vec, True)
        return merged_img

    def preprocess(self, face_vec):
        """
        调用_swap_face前的向量预处理
        :param face_vec: arcface输出的未经任何处理的512维向量
        :return: 处理后可传入的_swap_face向量
        """
        latent = face_vec  # 脸部向量：512
        latent /= np.linalg.norm(latent)
        latent = latent.reshape(1, -1)
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def _gen_face(self, align_section, norm_dst_vec):
        """
        面部替换
        :param align_section: 裁剪矫正后的面部区域
        :param norm_dst_vec: 预处理过的模特人脸向量
        :return:
        """
        blob = cv2.dnn.blobFromImage(align_section, 1.0 / self.input_std, self.input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: norm_dst_vec})[0]
        # pred = self._predict({self.input_names[0]: blob, self.input_names[1]: norm_dst_vec})

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]  # 转BGR
        return bgr_fake

    def _paste(self, target_img, bgr_fake, align_face, M):
        H, W = target_img.shape[0:2]
        fake_diff = bgr_fake.astype(np.float32) - align_face.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)  # fake_diff表示变脸前后的面部区域像素差(对三通道的差值的绝对值取平均)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0  # 将fake_diff边缘2圈像素置零(0为黑色), 用以忽略边缘部分的影响
        IM = cv2.invertAffineTransform(M)
        img_white = np.full((align_face.shape[0], align_face.shape[1]), 255, dtype=np.float32)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (W, H))  # borderValue=0.0
        img_white = cv2.warpAffine(img_white, IM, (W, H))  # borderValue=0.0
        fake_diff = cv2.warpAffine(fake_diff, IM, (W, H))  # borderValue=0.0
        img_white[img_white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)  # 返回img_mash中所有值为255的行&列的索引
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))  # 计算出mask方框的面积
        k = max(mask_size // 10, 10)
        # k = max(mask_size//20, 6)
        # k = 6
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        k = max(mask_size // 20, 5)
        # k = 3
        # k = 3
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255
        # img_mask = fake_diff
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
        fake_merged = fake_merged.astype(np.uint8)
        return fake_merged

    def swap(self, img, cur_face_kps, dst_face_vec, paste_back=True):
        """
        照片换脸
        :param img: 源图像
        :param cur_face_kps: 等待被替换的目标脸信息Face.kps关键点
        :param dst_face_vec: 通过w600k_r50.onnx人脸识别模型向量化后的模特人脸向量, (512,)
        :param paste_back: 变换后的脸是否融合回源图像
        :return: 融合图
        """
        dst_face_vec = self.preprocess(dst_face_vec)
        align_face, matrix = face_align.norm_crop2(img, cur_face_kps, self.input_size[0])  # 128
        bgr_fake = self._gen_face(align_face, dst_face_vec)
        if not paste_back:
            return bgr_fake, matrix
        else:
            # return self._paste(img, bgr_fake, align_face, matrix)
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - align_face.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2, :] = 0
            fake_diff[-2:, :] = 0
            fake_diff[:, :2] = 0
            fake_diff[:, -2:] = 0
            IM = cv2.invertAffineTransform(matrix)
            img_white = np.full((align_face.shape[0], align_face.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white[img_white > 20] = 255
            fthresh = 10
            fake_diff[fake_diff < fthresh] = 0
            fake_diff[fake_diff >= fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 10, 10)
            # k = max(mask_size//20, 6)
            # k = 6
            kernel = np.ones((k, k), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
            k = max(mask_size // 20, 5)
            # k = 3
            # k = 3
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2 * i + 1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            # img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged

    def swap_batch(self, img, faces_kps, dst_face_vecs, paste_back=True):
        aligned_faces, matrices, input_batch, norm_dst_vecs = [], [], [], []
        for i, face_kps in enumerate(faces_kps):
            align_face, matrix = face_align.norm_crop2(img, face_kps, self.input_size[0])  # 128
            aligned_faces.append(align_face)
            matrices.append(matrix)
            blob = cv2.dnn.blobFromImage(align_face, 1.0 / self.input_std, self.input_size,
                                         (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            input_batch.append(blob)
            norm_dst_vecs.append(self.preprocess(dst_face_vecs[i]))

        input_batch = np.concatenate(input_batch)
        norm_dst_vecs = np.concatenate(norm_dst_vecs)

        pred = self.session.run(self.output_names,
                                {self.input_names[0]: input_batch, self.input_names[1]: norm_dst_vecs})[0]
        # pred = self._predict_batch({self.input_names[0]: input_batch, self.input_names[1]: norm_dst_vecs})

        img_fakes = pred.transpose((0, 2, 3, 1))
        bgr_fakes = np.clip(255 * img_fakes, 0, 255).astype(np.uint8)[:, :, :, ::-1]  # 转BGR

        if not paste_back:
            return bgr_fakes, matrices
        else:
            merged_img = img
            for i, bgr_fake in enumerate(bgr_fakes):
                merged_img = self._paste(merged_img, bgr_fake, aligned_faces[i], matrices[i])
            return merged_img

    def swap_many(self, img, faces_kps, dst_face_vecs):
        """
        照片换脸
        :param img: 源图像
        :param faces_kps: 当前图中所有待替换的目标人脸关键点列表, list(target.kps)
        :param dst_face_vecs: w600k_r50.onnx向量化的所有对应的模特人脸向量列表, list(model.vector)
        :return: 融合图
        """
        for i, kps in enumerate(faces_kps):
            img = self.swap(img, kps, dst_face_vecs[i], paste_back=True)
        return img

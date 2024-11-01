import os
import pickle
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from tqdm import tqdm

from modules.face_module.face_analyze import Face, FaceAnalyzer
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.lipsync_module.musetalk.face_detection import FaceAlignment, LandmarksType

model_root = "D:/PycharmProjects/open_metahuman/models/lipsync_models"

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_module_dir = os.path.dirname(os.path.abspath(__file__))
config_file = f'{current_module_dir}/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = f'{model_root}/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0, 0.0, 0.0, 0.0)





model_root = Path('D:/PycharmProjects/open_metahuman/models/face_models')
detector = RetinaFaceOnnx(model_file=model_root / 'det_10g.onnx', device='cpu', det_size=(640, 640),
                          max_faces=5)  # 人脸检测
landmarker = Landmark3d68ONNX(model_file=model_root / '1k3d68.onnx', device='cpu')  # 关键点标记
processor = FaceAnalyzer()
processor.add_model(detector, 'detector')
processor.add_model(landmarker, 'landmarker')






def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）

    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for i, fb in enumerate(tqdm(batches)):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints

        # face_land_mark = keypoints[0].astype(np.int32)
        # for (x, y) in face_land_mark:
        #     cv2.circle(frames[i], (x, y), 2, (0, 255, 255), -1)  # 绘制每个关键点，黄色

        face_land_mark = keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)

        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))  # x1, y1, x2, y2

        # DEBUG:画图
        # for (x, y) in face_land_mark:
        #     cv2.circle(frames[i], (x, y), 2, (0, 255, 0), -1)  # 绘制每个关键点，绿色
        # cv2.circle(frames[i], (face_land_mark[29][0], face_land_mark[29][1]), 2, (0, 0, 255), -1)
        # cv2.circle(frames[i], (face_land_mark[28][0], face_land_mark[28][1]), 2, (255, 0, 0), -1)
        # cv2.circle(frames[i], (face_land_mark[30][0], face_land_mark[30][1]), 2, (255, 0, 0), -1)

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):  # int
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]

    print(
        "********************************************bbox_shift parameter adjustment**********************************************************")
    print(
        f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print(
        "*************************************************************************************************************************************")
    return coords_list, frames


def get_landmark_and_bbox2(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batches = frames

    coords_list = []
    landmarks = []

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for i, fb in enumerate(tqdm(batches)):
        faces: List[Face] = processor.apply(fb)

        if not faces:
            coords_list.append(coord_placeholder)
            continue

        face = faces[0]

        bbox = face.bbox.astype(np.int32)
        keypoints = face.landmark.astype(np.int32)

        face_land_mark = keypoints
        # for (x, y, z) in face_land_mark:
        #     cv2.circle(frames[i], (x, y), 2, (0, 255, 255), -1)  # 绘制每个关键点，黄色

        # DEBUG:画图
        # for (x, y, z) in face_land_mark:
        #     cv2.circle(frames[i], (x, y), 2, (0, 255, 0), -1)  # 绘制每个关键点，绿色
        # cv2.circle(frames[i], (face_land_mark[29][0], face_land_mark[29][1]), 2, (0, 0, 255), -1)
        # cv2.circle(frames[i], (face_land_mark[28][0], face_land_mark[28][1]), 2, (255, 0, 0), -1)
        # cv2.circle(frames[i], (face_land_mark[30][0], face_land_mark[30][1]), 2, (255, 0, 0), -1)

        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue

            half_face_coord = face_land_mark[29]  # np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30] - face_land_mark[29])[1]
            range_plus = (face_land_mark[29] - face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]  # 手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
            upper_bond = half_face_coord[1] - half_face_dist

            f_landmark = (
                np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]),
                np.max(face_land_mark[:, 1]))
            x1, y1, x2, y2 = f_landmark

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w, h = f[2] - f[0], f[3] - f[1]
                print("error bbox:", f)
            else:
                coords_list += [f_landmark]

    print(
        "********************************************bbox_shift parameter adjustment**********************************************************")
    print(
        f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print(
        "*************************************************************************************************************************************")
    return coords_list, frames



if __name__ == "__main__":
    img_list = ["D:/PycharmProjects/open_metahuman/app/results/obama2/00000000.png", "D:/PycharmProjects/open_metahuman/app/results/obama2/00000001.png", "D:/PycharmProjects/open_metahuman/app/results/obama2/00000002.png",
                "D:/PycharmProjects/open_metahuman/app/results/obama2/00000003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
    i = 0
    for bbox, frame in zip(coords_list, full_frames):
        cv2.imwrite(f'D:/PycharmProjects/open_metahuman/app/results/{i}.png', frame)
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)

        img = full_frames[i][y1:y2, x1:x2]
        cv2.imwrite(f'D:/PycharmProjects/open_metahuman/app/results/{i}_scrfd.png', img)
        i += 1
    print(coords_list)

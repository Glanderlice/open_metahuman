from PIL import Image
import numpy as np
import cv2

from modules.lipsync_module.musetalk.face_parsing import FaceParsing

fp = FaceParsing("D:/PycharmProjects/open_metahuman/models/lipsync_models")


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]
    return crop_box, s


def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)

    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.2):
    # print(image.shape)
    # print(face.shape)
    # 将图像从 BGR 转换为 RGB 格式, 使用切片操作将图像数组的最后一个维度反转
    body = Image.fromarray(image[:, :, ::-1])
    face = Image.fromarray(face[:, :, ::-1])

    x, y, x1, y1 = face_box
    # print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_position = (x, y)

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    # 输出人脸面部是mask掩膜
    mask_image = face_seg(face_large)

    cv2.imshow('img1', cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # 得到原来脸部的mask掩膜(相减后得到mask_image中(expand前)脸部的相对区域)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    cv2.imshow('img2', cv2.cvtColor(np.array(mask_small), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    mask_image = Image.new('L', ori_shape, 0)  # 构造一个全黑的灰度图
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    cv2.imshow('img3', cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_image = Image.fromarray(mask_array)

    cv2.imshow('img4', cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    # 把改变口型的脸覆盖到face_large上,实现面部口型的变换, 这里的覆盖是没有mask的, 因此可能出现新脸跟原脸不重叠的问题(因此需要特别注意裁剪下来进行计算的面部区域最好是完整的脸,特别是下巴)
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    # 把face_large通过mask融合到原图(body)上
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:, :, ::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    body = Image.fromarray(image[:, :, ::-1])

    x, y, x1, y1 = face_box
    # print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    face_large[y - y_s:y1 - y_s, x - x_s:x1 - x_s] = face

    mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image / 255).astype(np.float32)

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large, body[y_s:y_e, x_s:x_e], mask_image, 1 - mask_image)

    return body

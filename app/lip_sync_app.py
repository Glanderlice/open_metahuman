import argparse
import os
from pathlib import Path
from typing import List, Union

from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy

import shutil

from modules.face_module.face_analyze import FaceAnalyzer, Face
from modules.face_module.face_model.detectors import RetinaFaceOnnx
from modules.face_module.face_model.landmarkers import Landmark3d68ONNX
from modules.lipsync_module.musetalk.blending import get_image
from modules.lipsync_module.musetalk.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, \
    get_landmark_and_bbox2
from modules.lipsync_module.musetalk.utils import get_file_type, get_video_fps, load_all_model, datagen
from modules.lipsync_module.musetalk.whisper.audio2feature import Audio2Feature

# load model weights
audio_processor, vae, unet, pe = load_all_model("D:/PycharmProjects/open_metahuman/models/lipsync_models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)


@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    inference_config = OmegaConf.load(args.inference_config)

    video_path = inference_config[task_id]["video_path"]
    audio_path = inference_config[task_id]["audio_path"]
    bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename)  # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path,
                                        input_basename + ".pkl")  # only related to video input
    os.makedirs(result_img_save_path, exist_ok=True)

    # debug
    temp_img_save_path = os.path.join(args.result_dir, output_basename + '/temp')
    os.makedirs(temp_img_save_path, exist_ok=True)
    decode_img_save_path = os.path.join(args.result_dir, output_basename + '/decode')
    os.makedirs(decode_img_save_path, exist_ok=True)

    if args.output_vid_name is None:
        output_vid_name = os.path.join(args.result_dir, output_basename + ".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path, ]
        fps = args.fps
    elif os.path.isdir(video_path):  # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    else:
        raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

    # print(input_img_list)
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)

    i = 0
    input_latent_list = []
    for k, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(f"{temp_img_save_path}/{str(k).zfill(8)}.png", crop_frame)

        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    frame_list_cycle = frame_list
    coord_list_cycle = coord_list
    input_latent_list_cycle = input_latent_list
    # 将列表反向再与本身拼接(如果视频长度比音频短, 会从头复用视频, 这里采用"倒放拼接"是为了让连接处更平滑)
    # frame_list_cycle = frame_list + frame_list[::-1]
    # coord_list_cycle = coord_list + coord_list[::-1]
    # input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                     dtype=unet.model.dtype)  # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            cv2.imwrite(f"{decode_img_save_path}/{str(i).zfill(8)}_gen.png", res_frame)
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))  # 将256x256缩放到人脸bbox的尺寸
            cv2.imwrite(f"{decode_img_save_path}/{str(i).zfill(8)}_rez.png", res_frame)
        except:
            #                 print(bbox)

            continue

        combine_frame = get_image(ori_frame, res_frame, bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)

    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    # shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')

    args = parser.parse_args()
    args.use_float16 = True
    main(args)


def extract_face_feature(video_path, bbox_shift, frame_save_path, coord_save_path, use_saved_coord=True):
    os.makedirs(frame_save_path, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {frame_save_path}/%08d.png"
    os.system(cmd)
    input_img_list = sorted(glob.glob(os.path.join(frame_save_path, '*.[jpJP][pnPN]*[gG]')))
    fps = get_video_fps(video_path)

    if coord_save_path and os.path.exists(coord_save_path) and use_saved_coord:
        print("using extracted coordinates")
        with open(coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    return coord_list, frame_list, fps


def extract_audio_feature(audio_path, video_fps):
    """
    提取音频特征：whisper_chunks
    :param audio_path:
    :param video_fps:
    :return:
    """
    audio_processor = Audio2Feature(model_path=f"{model_root}/whisper/tiny.pt")
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=video_fps)
    return whisper_chunks


def lip_sync(whisper_chunks, frame_list, coord_list, save_as, cycle=False):
    # 1.裁剪人脸区域crop_frame并缩放到256x256, 并通过VAE编码到隐空间->input_latent_list
    input_latent_list = []
    for k, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # cv2.imwrite(f"{temp_img_save_path}/{str(k).zfill(8)}.png", crop_frame)  # DEBUG

        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    if cycle:
        # 将列表反向再与本身拼接(如果视频长度比音频短, 会从头复用视频, 这里采用"倒放拼接"是为了让连接处更平滑)
        frame_list = frame_list + frame_list[::-1]
        coord_list = coord_list + coord_list[::-1]
        input_latent_list = input_latent_list + input_latent_list[::-1]

    # 2.音频驱动口型同步
    video_num = len(whisper_chunks)
    batch_size = 8  # 这里取的是源码中默认值
    gen = datagen(whisper_chunks, input_latent_list, batch_size)

    res_frame_list = []  # 用于存储VAE解码后的面部图像
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                     dtype=unet.model.dtype)  # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)  # 音频位置编码

        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        # 音频向量+人脸(Ref+Masked)隐空间特征图=>同步人脸隐特征图latent
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        # 解码：latent->RGB, 然后RGB->BGR
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    # 3.融合到原视频帧
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list[i % (len(coord_list))]
        ori_frame = copy.deepcopy(frame_list[i % (len(frame_list))])
        x1, y1, x2, y2 = bbox
        try:
            # cv2.imwrite(f"{decode_img_save_path}/{str(i).zfill(8)}_gen.png", res_frame)
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))  # 将256x256缩放到人脸bbox的尺寸
            # cv2.imwrite(f"{decode_img_save_path}/{str(i).zfill(8)}_rez.png", res_frame)
        except:
            #                 print(bbox)

            continue

        combine_frame = get_image(ori_frame, res_frame, bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)

    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {save_as}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    # shutil.rmtree(result_img_save_path)
    print(f"result is save to {save_as}")


if __name__ == '__main__':

    video_path = "D:/PycharmProjects/open_metahuman/data/video/dh_2.mp4"
    audio_path = "D:/PycharmProjects/open_metahuman/data/audio/taiwan.WAV"
    use_float16 = True

    model_root = "D:/PycharmProjects/open_metahuman/models/lipsync_models"
    audio_processor, vae, unet, pe = load_all_model(str(model_root))

    if use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    print('end')

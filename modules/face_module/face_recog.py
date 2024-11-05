import heapq
import os
import pickle
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from modules.face_module.face_analyze import Face
from modules.face_module.face_model.embedders import ArcFaceOnnx
from tool.file_util import clear_dir
from tool.image_helper import cal_overlap


class FaceMeta:
    """
    person: 这是谁的脸
    vector: 脸向量(仅对应特定模型,例如facenet与arcface的向量完全不一样)
    image:  脸部图像(从原图裁剪->对齐)
    pose: 脸部姿态
    """

    def __init__(self, pid, vec, image=None, pose=None, position=None):
        self.pid = pid
        self.vec = vec
        self.image = image
        self.pose = pose

    def __reduce__(self):
        return self.__class__, (self.pid, self.vec, self.image, self.pose)  # 反序列化

    def __str__(self):
        return f'{self.pid}:vector={0 if self.vec is None else 1},image={0 if self.image is None else 1}'

    def __repr__(self):
        return self.__str__()


def _ann_top_1(vector: np.ndarray, vector_index: Dict[str, FaceMeta], ranges: List[str] = None):
    """for循环搜索人脸向量库"""
    find_person, max_sim = None, 0
    ranges = ranges if ranges else vector_index
    for p in ranges:
        face_data = vector_index.get(p, None)
        if face_data is not None:
            sim = _cosine_sim(vector, face_data.vec)
            ret = _classify(sim)
            if ret > 0 and sim > max_sim:
                find_person, max_sim = face_data, sim  # 找到了
    return [(find_person, max_sim)]  # 只会返回最相似的(人,相似值),或者(None,阈值)


def _ann_top_k(vector: np.ndarray, k: int, vector_index: Dict[str, FaceMeta], ranges: List[str] = None) -> List[
    Tuple[object, float]]:
    """
    搜索人脸向量库，返回最相似的 topk 个结果。
    Args:
        vector (np.ndarray): 查询的人脸向量。
        k (int): 返回的相似结果的个数。
    Returns:
        List[Tuple[object, float]]: 最相似的 topk 个结果，每个结果为一个元组 (face_data, sim), 其中 face_data 是人脸数据对象，sim 是相似度。
    """
    # 使用优先队列维护top k个相似结果
    top_k_heap = []
    ranges = ranges if ranges else vector_index
    for p in ranges:
        face_data = vector_index.get(p, None)
        sim = _cosine_sim(vector, face_data.vec)
        ret = _classify(sim)
        if ret > 0:
            # 将结果加入堆中，如果堆的大小超过top k，则弹出最小的元素
            heapq.heappush(top_k_heap, (sim, face_data))  # heapq默认使用元组的第一个元素来排序, 默认为最小堆
            if len(top_k_heap) > k:
                heapq.heappop(top_k_heap)

    # 将结果按相似度从高到低排序并返回
    top_k_heap.sort(reverse=True, key=lambda x: x[0])
    return [(face_data, sim) for sim, face_data in top_k_heap]


class SimpleFaceDB:
    """
    人脸数据库, 支持最简单的功能：
    1.添加：id,向量,图片,可根据姿态好坏进行覆盖
    2.查询/删除：根据id调用get/remove返回或删除人脸数据
    3.搜索：根据向量进行ANN搜索,返回topK个最相似项
    """

    def __init__(self, update_scheme='pose'):
        """
        人脸数据库(in memory)
        :param update_scheme: None/'last'/'pose', 默认None即保留第一张人脸, last保留最后的人脸, pose保留姿态最好的人脸
        """
        self.face_index = {}
        self.counter = {}
        self.scheme = update_scheme

    def add(self, pid, vector, image, pose=None):
        """
        尝试向数据库里添加人脸, 如果已存在则根据更新策略scheme来判断
        :param pid: 人物ID
        :param vector: 人脸向量(与某个模型绑定)
        :param image: (矫正后的)人脸
        :param pose: 人脸姿态数组
        :return: 0不变, 1新增, 2替换
        """
        # 记录次数
        self.counter[pid] = self.counter.get(pid, 0) + 1

        face = FaceMeta(pid, vector, image, pose)
        ret = 0  # 未变更
        if pid not in self.face_index:
            self.face_index[pid] = face
            ret = 1  # 新增数据
        else:
            if self.scheme == 'pose':
                exist_face: FaceMeta = self.face_index.get(pid)
                if pose is not None and exist_face.pose is not None:
                    norm1 = np.linalg.norm(pose[:2])  # [:2]表示不要Roll值,因为输入的是矫正过的脸
                    norm2 = np.linalg.norm(exist_face.pose[:2])
                    if norm1 < norm2:
                        self.face_index[pid] = face
                        ret = 2  # 替换原数据(姿态更好)
                    else:
                        ret = -1  # 放弃替换(姿态更差)
                else:
                    print(f'face pose missing, query pose = {pose}, exist pose = {exist_face.pose}')
            elif self.scheme == 'last':
                self.face_index[pid] = face
                ret = 2
        return ret

    def ann_search(self, vector: np.ndarray, k: int = 1, ranges: List[str] = None) -> List[Tuple[Any, float]]:
        assert k > 0
        if k == 1:
            return _ann_top_1(vector, self.face_index, ranges)
        else:
            return _ann_top_k(vector, k, self.face_index, ranges)

    def rename(self, old_name: str, new_name: str):
        """人脸重命名：old_name -> new_name"""
        if old_name in self.face_index and new_name and new_name not in self.face_index:
            face_data: FaceMeta = self.face_index[old_name]
            del self.face_index[old_name]
            face_data.pid = new_name
            self.face_index[new_name] = face_data
            return True
        else:
            return False

    def get(self, pid):
        """通过person_id返回人脸数据"""
        return self.face_index.get(pid, None)

    def remove(self, pid):
        """通过person_id删除人脸数据"""
        if pid and pid in self.face_index:
            del self.face_index[pid]

    def export_face_images(self, save_dir):
        """将所有人脸导出到文件夹: 以人名命名"""
        clear_dir(save_dir)
        for name, face in self.face_index.items():
            cv2.imwrite(os.path.join(save_dir, f'{name}.jpg'), face.image, [cv2.IMWRITE_JPEG_QUALITY, 75])

    def clear(self):
        self.face_index.clear()
        self.counter.clear()

    def save_db(self, db_path):
        """人脸数据序列化到文件"""
        if self.face_index:
            all_data = {"index": self.face_index, "count": self.counter}
            # 后缀统一改为.pkl
            filepath, extension = os.path.splitext(db_path)
            filepath = f'{filepath}.pkl'
            if extension != '.pkl':
                print(f'db path must end with .pkl, auto changed to {filepath}')

            if os.path.isfile(filepath):  # 覆盖提醒
                print(f'db {filepath} already exists, will be overwritten.')

            with open(filepath, 'wb') as file:
                pickle.dump(all_data, file)
            print(f'all face data saved to {filepath}')
        else:
            print('no data in db to save')

    def load_db(self, db_path):
        """从文件反序列化人脸数据"""
        self.clear()  # 先清理掉缓存

        loaded: bool = False
        if os.path.isfile(db_path):
            _, extension = os.path.splitext(db_path)
            if extension != '.pkl':
                print(f'db path {db_path} is not end with .pkl, try to load')
            # 从文件加载并反序列化字典
            with open(db_path, 'rb') as file:
                loaded_dict = pickle.load(file)
                if loaded_dict:
                    self.face_index, self.counter = loaded_dict['index'], loaded_dict['count']
                    loaded = True if self.face_index else False
                    print(f'{len(self.face_index)} data load from {db_path}')
                else:
                    print(f'no data load from {db_path}')
        else:
            print(f'db file {db_path} not exist')

        return loaded


class FaceRecognizer:
    """
    人脸识别：默认绑定一个内存级的人脸数据库facedb, 和一个面部追踪tracker
    """

    def __init__(self, embed_model: ArcFaceOnnx, facedb: SimpleFaceDB):  # glintr100.onnx'
        self.embed_model = embed_model
        self.input_size = embed_model.input_size
        self.facedb = facedb
        self.tracker = deque(maxlen=4)
        self.person_idx = 0

    def learn(self, face: Face):
        """
        将人脸入库, 如果已存在则返回False
        :param face: 人脸数据
        :return:
        """

        face_data, score = self._search_by_vector(face)  # 输入人脸的向量, 库中找到的人脸数据, 向量相似值(与距离公式有关)
        if face_data:
            assert face_data.pid is not None
            person = face_data.pid
        else:  # 未找到->发现新人脸,插入人脸库
            person = f'face_{self.person_idx}'
            self.person_idx += 1

        # 录入人脸
        state = self.facedb.add(person, face.vec, face.face_img, face.pose)
        return state

    def _track(self, det_box, frame_idx, iou_thresh, dst_thresh):
        """输入det_box必须是对角点np.array(x1,y1,x2,y2), shape=(4,)"""
        max_iou, dst, find = 0, 100, None
        for frame_i, boxes in reversed(self.tracker):
            if frame_i < frame_idx:
                for box, person in boxes:
                    iou, distance = cal_overlap(det_box, box)  # 重合度
                    if iou >= iou_thresh and distance <= dst_thresh:
                        max_iou, dst, find = iou, distance, person
                        iou_thresh = max_iou
        return max_iou, dst, find

    def _search_by_track(self, det_box, frame_idx):
        iou, dst, tracked_pid = self._track(det_box, frame_idx, 0.90, 10)  # 追踪缓存的识别框
        return tracked_pid

    def vectorize(self, face: Face, inplace=True):
        face_vec = face.vec
        if face_vec is None:
            align_face: np.ndarray = face.face_img  # 对齐后的人脸, size未知, 但必须是正方形
            h, w = align_face.shape[:2]
            if h != w:
                raise ValueError(f'face.H==face.W is required {h, w}')
            if (w, h) != self.input_size:
                align_face = cv2.resize(align_face, self.input_size)  # 输入人脸图像缩放(W,H)
            face_vec = self.embed_model.vectorize(align_face)[0]  # 该接口返回(batch, 512), 此处强行取第一个
            if inplace:
                face.vec = face_vec
        return face_vec

    def _search_by_vector(self, face: Face, ranges: List[str] = None):
        face_vec = self.vectorize(face, inplace=True)
        face_data, sim = self.facedb.ann_search(face_vec, 1, ranges)[0]
        return face_data, sim

    def _verify(self, face: Face, person: str):
        ret = -1
        if person:
            face_data = self.facedb.get(person)
            if face_data is not None:
                face_vec = self.vectorize(face, inplace=True)
                sim = _cosine_sim(face_vec, face_data.vec)
                ret = _classify(sim)
        return ret

    # @timing(interval=60)
    def search(self, face: Face, frame_idx: int = None, persons: List[str] = None):
        """
        联合识别：先使用检测框追踪track, 再对不那么确定的使用人脸识别search
        :param face: 人脸
        :param frame_idx: 当前帧数, 用于更新deque
        :param persons: 缩小搜索范围
        :return: 记忆中的人物
        """
        # 1.面部追踪(较快)
        det_box = face.bbox
        if det_box.shape == (4, 2):
            det_box = det_box[[0, 2]].ravel()  # 先变成对角点, 然后拉平成shape=(4,)
        target = self._search_by_track(det_box, frame_idx)  # 与前几帧的人脸位置最相近的人

        if target:
            if self._verify(face, target) < 0:
                print(f'同一检测框中追踪({target})与识别结果不一致')
                target = None

        if target is None:
            # 检索(较慢):人脸识别(搜人脸库)
            face_data, sim = self._search_by_vector(face, None)
            if face_data:  # 人脸识别成功
                target = face_data.pid

        if target and (persons is None or target in persons):
            self._update_tracker(det_box, target, frame_idx)

        return target

    def _update_tracker(self, det_box, person, frame_idx=None):
        """输入 det_box 必须是对角点 np.array(x1, y1, x2, y2), shape=(4,)"""
        dq = self.tracker

        # 检查是否需要创建新的队尾元素
        if not dq or (frame_idx is not None and 0 <= frame_idx != dq[-1][0]):
            last = []
            dq.append((frame_idx, last))  # 新增队尾元素
        else:
            last = dq[-1][1]  # 获取现有的队尾元素列表

        last.append((det_box, person))  # 追加新数据

    def most_frequent_faces(self, k: Optional[int] = None):
        """
        返回出现频率最高的n个人脸
        :param k: 返回多少个, 默认None为返回全部
        :return: list[tuple[人脸数据, 出现次数]]
        """
        counter: Dict[str, int] = self.facedb.counter
        if not counter:
            return None
        sorted_keys = sorted(counter, key=lambda k: counter[k], reverse=True)  # 降序
        if k and k > 0:
            sorted_keys = sorted_keys[:k]
        top_k = [(self.facedb.get(key), counter.get(key)) for key in sorted_keys]

        return top_k

    def clear(self):
        self.tracker.clear()


def _cosine_sim(vec1, vec2):
    vec1 = vec1.ravel()
    vec2 = vec2.ravel()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def _classify(sim, model='arcface'):
    """
    根据人脸相似度得出结论：
    -1 表示不同 (different)
     0 表示相似 (alike)
     1 表示相同 (same)
    """
    if model == 'arcface':
        if sim < 0.2:
            return -1
        elif sim < 0.28:
            return 0
        else:
            return 1
    else:
        raise NotImplementedError

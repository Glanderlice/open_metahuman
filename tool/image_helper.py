import numpy as np


def reverse_channel(img):
    """
    将输入图片的channel进行反转, 实现BGR(opencv默认顺序)->RGB或RGB->BGR
    :param img: 形状shape=[H,W,C=3]的图片, C即channel
    :return: 反转后的图片
    """
    return img[:, :, ::-1]


def cal_iou(rect1: np.ndarray | list[tuple], rect2: np.ndarray | list[tuple]):
    """
    计算两矩形(左上,右下)的IOU(Intersection over Union)面积交并比, 无相交则返回0
    :param rect1: [lt_x,lt_y,rb_x,rb_y], 其中lt表示left top, rb表示right bottom
    :param rect2: 同rect1
    :return:
    """
    # 计算两个子区域的矩形框的左上角和右下角坐标
    # rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
    lt_x1, lt_y1, rb_x1, rb_y1 = rect1[0], rect1[1], rect1[2], rect1[3]
    # rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]
    lt_x2, lt_y2, rb_x2, rb_y2 = rect2[0], rect2[1], rect2[2], rect2[3]
    if rb_x1 > lt_x2 and lt_x1 < rb_x2 and rb_y1 > lt_y2 and lt_y1 < rb_y2:
        # 计算重叠矩形的面积
        overlap_area = (min(rb_x1, rb_x2) - max(lt_x1, lt_x2)) * (min(rb_y1, rb_y2) - max(lt_y1, lt_y2))  # 交叉面积
        union_area = (rb_x1 - lt_x1) * (rb_y1 - lt_y1) + (rb_x2 - lt_x2) * (rb_y2 - lt_y2) - overlap_area  # 两面积和-交叉面积
        iou = overlap_area / union_area  # Intersection over Union
        return round(iou, 4)  # 只精确到小数点后2位
    else:
        return 0.


def cal_overlap(box1: np.ndarray | list[np.ndarray], box2: np.ndarray | list[np.ndarray]):
    """
    计算两个子区域IOU和中心距离。
    Args:
    box1: 第一个子区域的左上右下顶点坐标 [x1,y1,x2,y2]
    box2: 第二个子区域的左上右下顶点坐标
    Returns:
    overlap_area (float): IOU, 如果不重叠则返回0
    distance (float): 两个子区域中心点的距离
    """
    iou = cal_iou(box1, box2)
    # 计算中心点之间的距离
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2], dtype=np.float32)
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2], dtype=np.float32)
    distance = np.linalg.norm(center1 - center2)
    # distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)  # **2** 不能复制粘贴
    return iou, distance

import cv2

# 打开摄像头，参数 0 通常表示默认摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 逐帧捕获视频
    ret, frame = cap.read()

    # 如果没有读取到帧，结束循环
    if not ret:
        print("无法接收视频帧")
        break

    # 显示帧
    cv2.imshow('Camera', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

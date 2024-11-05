from pathlib import Path

import onnxruntime as ort
import onnx

# log level:
# 0: ORT_LOGGING_LEVEL_VERBOSE
# 1: ORT_LOGGING_LEVEL_INFO
# 2: ORT_LOGGING_LEVEL_WARNING
# 3: ORT_LOGGING_LEVEL_ERROR
# 4: ORT_LOGGING_LEVEL_FATAL
ort.set_default_logger_severity(3)


def build_session(onnx_model_path, device='cpu', verify=False):
    """
    使用device加载onnx格式模型, 默认device=cpu, 默认不进行格式检查verify=False
    支持的设备代号: [TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]
        1) cpu->CPUExecutionProvider
        2) cuda/gpu->CUDAExecutionProvider
        3) tensorrt->TensorrtExecutionProvider
    """
    # 'cpu','cuda','gpu','tensorrt', 映射成onnxruntime的device_name
    alias = device.lower()
    if alias == 'cpu':
        device = 'CPUExecutionProvider'
    elif alias == 'cuda' or alias == 'gpu':
        device = 'CUDAExecutionProvider'
    elif alias == 'tensorrt':
        device = 'TensorrtExecutionProvider'
    if device not in ort.get_available_providers():
        raise ValueError(f'{device} is not available in onnxruntime')
    # verify onnx model
    if verify:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        del onnx_model

    return ort.InferenceSession(onnx_model_path, providers=[device])


def onnx_cuda_test():
    """如果报错则无法正常调用cuda进行推理加速,可能是cudnn库没有被正确安装和配置"""
    print(ort.__version__)
    print(ort.get_device())  # 如果得到的输出结果是GPU，所以按理说是找到了GPU的

    app_root = Path(__file__).parent.parent
    print(app_root / "models/face_models/det_10g.onnx")
    ort_session = ort.InferenceSession(app_root / "models/face_models/det_10g.onnx",
                                       providers=['CUDAExecutionProvider'])
    print(ort_session.get_providers())


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    onnx_cuda_test()

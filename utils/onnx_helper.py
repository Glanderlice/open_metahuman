import onnxruntime as ort
import onnx


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
        onnx_model = onnx.load("super_resolution.onnx")
        onnx.checker.check_model(onnx_model)
        del onnx_model

    return ort.InferenceSession(onnx_model_path, providers=[device])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

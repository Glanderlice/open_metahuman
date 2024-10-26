from GPUtil import GPUtil, GPU


def _check_gpu_usage(gpu: GPU):
    usage = round(gpu.memoryUtil, 3)
    used_gb = round(gpu.memoryUsed / 1024, 3)
    total_gb = round(gpu.memoryTotal / 1024, 3)
    free_gb = round(gpu.memoryFree / 1024, 3)
    return usage, used_gb, total_gb, free_gb


def check_gpu(index=None):
    gpus = GPUtil.getGPUs()
    gpu_usage, gpu_used_gb, gpu_total_gb, gpu_free_gb = 0, 0, 0, 0
    if gpus:
        if index is None:
            for i, gpu in enumerate(gpus):
                usage, used_gb, total_gb, free_gb = _check_gpu_usage(gpu)
                gpu_usage += usage
                gpu_used_gb += used_gb
                gpu_total_gb += total_gb
                gpu_free_gb += free_gb
                print(f'GPU[{i}]的使用率 {round(usage * 100, 1)}%: {used_gb}/{total_gb} GB, 剩余 {free_gb} GB')
            if len(gpus) > 1:
                gpu_usage = round(gpu_usage / len(gpus), 3)
        else:
            assert 0 <= index < len(gpus)
            gpu_usage, gpu_used_gb, gpu_total_gb, gpu_free_gb = _check_gpu_usage(gpus[index])
            print(f'GPU[{index}]的使用率 {round(gpu_usage * 100, 1)}%: {gpu_used_gb}/{gpu_total_gb} GB, 剩余 {gpu_free_gb} GB')
    else:
        print('没有发现GPU')
    return gpu_usage, gpu_used_gb, gpu_total_gb, gpu_free_gb

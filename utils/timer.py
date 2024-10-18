import time
from collections import OrderedDict

from GPUtil import GPUtil


def timing(interval: [int, None] = None, gpu: bool = False):
    """
    计时器装饰器：快速打印某次调用的耗时
    :param interval: 打印间隔, 即在一次打印后下次再打印的次数间隔, 默认值None表示每次都打印
    :param gpu: 是否计算GPU使用
    :return:
    """
    assert not interval or interval >= 0
    call_cnt = 0

    def wrapper(func):
        def wrapped_function(*args, **kwargs):
            nonlocal call_cnt  # 使用nonlocal关键字访问外部函数的call_count
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            exe_time = end_time - start_time
            call_cnt += 1  # 每次调用增加计数
            if not interval or call_cnt % interval == 0:
                if gpu:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        for i, _gpu in enumerate(gpus):
                            print(f'[GPU usage]: GPU[{i}] used {round(_gpu.memoryUtil*100,1)}%: {round(_gpu.memoryUsed/1024,3)}/{round(_gpu.memoryTotal/1024,3)} GB Used, {round(_gpu.memoryFree/1024,3)} GB Free')
                print(f"[time usage]: function='{func.__name__}' took {exe_time:.3f} seconds to execute.")
            return result  # 返回函数的结果，如果有的话

        return wrapped_function

    return wrapper


class Timing:
    """计时器装饰器类：可以累计总耗时,总调用次数, 并自定义输出总报表"""

    _instance = None  # 单例模式

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.timings = None
            cls._instance.counter = None
        return cls._instance

    def __init__(self):
        if self.timings is None:
            self.timings = OrderedDict()
        if self.counter is None:
            self.counter = dict()

    # 注意: 单例模式不用__init__, 实例化先调用__new__然后会再调用__init__, 应避免__init__方法内部意外修改某些成员变量
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            exe_time = end_time - start_time

            func_name = func.__name__

            all_time = self.timings.get(func_name, 0)
            self.timings[func_name] = all_time + exe_time

            all_cnt = self.counter.get(func_name, 0)
            self.counter[func_name] = all_cnt + 1

            return result

        return wrapper

    def clear(self):
        self.timings = OrderedDict()
        self.counter = dict()

    def get_all(self):
        return self.timings, self.counter

    def report(self, clear=True):
        timings, counter = self.get_all()
        print('[timing] - report:')

        keys = timings.keys()

        max_key_len = max(len(str(k)) for k in keys)

        for k in keys:
            all_time = timings.get(k)
            all_cnt = counter.get(k)
            k = str(k).ljust(max_key_len)  # 给key统一加了单引号
            print(f"\t{k} = {all_time / all_cnt:.3f} sec/call, totally {all_cnt} calls in {all_time:.3f} seconds.")
        print('[timing] - end')
        if clear:
            self.clear()


if __name__ == '__main__':
    tim1 = Timing()
    tim2 = Timing()
    tim3 = Timing()

    print(tim1 is tim2)
    print(tim1 is tim3)

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor

DEFAULT_THREADS = 4  # 默认线程数

MAX_THREADS_ALLOWED = 16  # 允许的最大线程数

lock = threading.Lock()


class MultiThread:
    _instance = None
    _executor = None
    _max_workers = None

    def __new__(cls, max_workers: int | None = None):
        """
        获取单例线程池：
        1.当max_workers=None, 表示获取当前单例, 若不存在则初始化max_workers=4的线程池再返回
        2.当max_workers=int有值, 若它等于当前单例的线程数则返回当前单例, 若不相等则重新初始化max_workers=int的新线程池再返回
        :param max_workers: 最大线程数,范围[1,32]
        """
        assert max_workers is None or isinstance(max_workers, int) and 1 <= max_workers <= MAX_THREADS_ALLOWED
        if cls._instance is None:
            cls._instance = super(MultiThread, cls).__new__(cls)
            cls._build(max_workers)
            atexit.register(cls.shutdown)  # 注册后, 在程序手动/自动终止时将调用, 释放线程池
        elif max_workers is not None and max_workers != cls._max_workers:
            cls.shutdown()  # 先关闭原线程池
            cls._build(max_workers)
        return cls._instance

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    @classmethod
    def _build(cls, max_workers):
        with lock:
            cls._max_workers = DEFAULT_THREADS if max_workers is None else max_workers
            cls._executor = ThreadPoolExecutor(max_workers=cls._max_workers)
            print(f'[info] multi-thread pool size = {cls._max_workers} built.')

    @classmethod
    def shutdown(cls):
        print('[info] multi-thread pool auto shutdown.')  # 自动关闭线程池
        if cls._executor:
            cls._executor.shutdown()

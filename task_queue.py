# task_queue.py
import queue
import threading

class TaskQueue:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.q = queue.Queue(maxsize=200)
        self.workers = []
        for i in range(4):  # 4个worker，GPU不卡
            t = threading.Thread(target=self._worker, daemon=True, name=f"AI-Worker-{i}")
            t.start()
            self.workers.append(t)
        print(f"[任务队列] 已启动 {len(self.workers)} 个推理线程")
        self.initialized = True

    def _worker(self):
        while True:
            try:
                func, args, kwargs = self.q.get()
                func(*args, **kwargs)
            except Exception as e:
                print(f"[Worker崩溃] {e}")
                import traceback; traceback.print_exc()
            finally:
                self.q.task_done()

    def put(self, func, *args, **kwargs):
        self.q.put((func, args, kwargs))

def get_task_queue():
    return TaskQueue()
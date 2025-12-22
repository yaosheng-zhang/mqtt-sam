# mqtt.py
import json
import time
import threading
import paho.mqtt.client as mqtt
from task_queue import get_task_queue
from system_monitor import SystemMonitor
import numpy as np

class MQTTClient:
    def __init__(self, config):
        self.config = config
        self.task_queue = get_task_queue()
        self.handlers = {}
        self.is_busy = False
        self.busy_lock = threading.Lock()

        self.client_id = f"sam3_{int(time.time())}"
        # 改用 MQTTv311 (更通用,大多数 Broker 都支持)
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)

        if config.mqtt_username:
            self.client.username_pw_set(config.mqtt_username, config.mqtt_password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        self.client.on_disconnect = self.on_disconnect  # 添加断开连接回调

        self.heartbeat_running = False
        self.publish_acked = {}
        self.publish_lock = threading.Lock()

    def set_busy(self, busy: bool):
        with self.busy_lock:
            self.is_busy = busy

    def get_device_status(self):
        with self.busy_lock:
            return 2 if self.is_busy else 1

    def register_handler(self, handler):
        topic = handler.subscribe_topic
        self.handlers[topic] = handler
        print(f"[MQTT] 注册 → {handler.get_name():12} | {topic}")

    def start(self):
        print(f"[MQTT] 连接 {self.config.mqtt_broker}:{self.config.mqtt_port}")
        self.client.connect(self.config.mqtt_broker, self.config.mqtt_port, 60)
        self.client.loop_start()
        self._start_heartbeat()

        print("[MQTT] 服务运行中... (Ctrl+C 退出)")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._stop_heartbeat()
        self.client.loop_stop()
        self.client.disconnect()
        print("[MQTT] 已停止")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] 连接成功！订阅主题：")
            for topic in self.handlers:
                client.subscribe(topic)
                print(f"   → {topic}")
        else:
            print(f"[MQTT] 连接失败 code={rc}")

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        if rc != 0:
            print(f"[MQTT] 意外断开连接 (code={rc})，正在重连...")
        else:
            print("[MQTT] 正常断开连接")

    def on_message(self, client, userdata, msg):
        """关键：只做解析 + 入队，<5ms 返回！"""
        try:
            data = json.loads(msg.payload)
            parts = msg.topic.split('/')
            if len(parts) < 4: return
            dev_id = parts[3]

            handler = self._match_handler(msg.topic)
            if handler:
                self.set_busy(True)
                self.task_queue.put(
                    self._run_task,
                    handler, dev_id, data, self.publish
                )
        except Exception as e:
            print(f"[MQTT] 解析失败: {e}")

    def _run_task(self, handler, dev_id, data, publish):
        try:
            handler.on_message(dev_id, data, publish)
        finally:
            self.set_busy(False)

    def _match_handler(self, topic):
        if topic in self.handlers:
            return self.handlers[topic]
        import re
        for pattern, h in self.handlers.items():
            if re.match(pattern.replace('+', '[^/]+').replace('#', '.*') + '$', topic):
                return h
        return None

    def publish(self, topic, data, wait_for_publish=False, qos=1):
       
        def convert(o):
            if isinstance(o, (np.integer, np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.floating, np.float64, np.float32)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"不能序列化: {type(o)}")

        payload = json.dumps(data, ensure_ascii=False, default=convert)
        info = self.client.publish(topic, payload, qos=qos)

    def on_publish(self, client, userdata, mid):
        with self.publish_lock:
            if mid in self.publish_acked:
                self.publish_acked[mid].set()

    def _start_heartbeat(self):
        self.heartbeat_running = True
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def _heartbeat_loop(self):
        while self.heartbeat_running:
            time.sleep(self.config.mqtt_heartbeat_interval)
            status = SystemMonitor.get_system_status()
            dev_status = self.get_device_status()
            payload = {
                "cmd": "devstate",
                "gateway": self.config.mqtt_dev_id,
                "data": {
                    "cpuUsage": round(status['cpu_usage']*100, 1),
                    "memoryUsage": round(status['memory_usage']*100, 1),
                    "gpuUsage": round(status['gpu_usage']*100, 1),
                    "devicestatus": dev_status
                },
                "timestamp": int(time.time()*1000)
            }
            self.publish(f"{self.config.mqtt_heartbeat_topic}/{self.config.mqtt_dev_id}", payload)
            print(f"[心跳] {'忙' if dev_status==2 else '闲'} | CPU {status['cpu_usage']*100:.0f}% | GPU {status['gpu_usage']*100:.0f}%")

    def _stop_heartbeat(self):
        self.heartbeat_running = False
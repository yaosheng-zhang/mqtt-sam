# handlers/base_handler.py
from abc import ABC, abstractmethod

class MessageHandler(ABC):
    def __init__(self, ai_engine, config):
        self.ai_engine = ai_engine
        self.config = config

    @property
    @abstractmethod
    def subscribe_topic(self): pass

    @abstractmethod
    def get_name(self): pass

    @abstractmethod
    def on_message(self, dev_id, data, publish): pass
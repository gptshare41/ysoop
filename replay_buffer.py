import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        """버퍼를 초기화합니다."""
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        """경험을 버퍼에 추가합니다."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """버퍼에서 무작위로 배치를 샘플링합니다."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """버퍼의 현재 크기를 반환합니다."""
        return len(self.buffer)

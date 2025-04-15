from abc import ABC

from src.model.plant import Plant


class Tomato(Plant, ABC):
    def __init__(self, config):
        self.config = config
    pass
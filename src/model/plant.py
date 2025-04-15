from abc import ABC, abstractmethod
from typing import List, Tuple


class Plant(ABC):

    @abstractmethod
    def get_leaf_arrangement_coords(self, center, num_leaves=25) -> List[Tuple]:
        pass

    @abstractmethod
    def get_angle_of_rotation_for_coords(self, center, coords) -> List:
        pass

    @abstractmethod
    def get_leaves(self, disease_type, angle, scale_factor):
        pass
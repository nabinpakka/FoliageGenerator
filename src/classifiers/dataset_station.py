import numpy as np

import paddle as paddle
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import Compose, Resize, Transpose, CenterCrop


class StationType(paddle.io.Dataset):
    def __init__(self, mode='train'):
        super(StationType, self).__init__()

        train_path = 'path_to_data/train'
        test_path = 'path_to_data/test'
        val_path = 'path_to_data/val'

        # composeTrans = Compose([Transpose()])
        composeTrans = Compose([Resize(size=(1024, 1024)), Transpose()])
        # composeTrans = Compose([CenterCrop((256 - 100, 256 - 100)), Transpose()])

        folderTrain = DatasetFolder(train_path, transform=composeTrans)
        folderVal = DatasetFolder(val_path, transform=composeTrans)
        folderTest = DatasetFolder(test_path, transform=composeTrans)

        self.mode = mode
        if self.mode == 'train':
            self.data = folderTrain
        elif self.mode == 'val':
            self.data = folderVal
        elif self.mode == 'test':
            self.data = folderTest

    def __getitem__(self, index):
        try:
            data = np.array(self.data[index][0]).astype('float32') / 255
            label = np.array([self.data[index][1]]).astype('int64')
            return data, label
        except (OSError, IOError) as e:
            print(f"Corrupted data at index {index}: {e} ")
            # Option 1: Skip to next valid index
            if index + 1 < len(self.data):
                return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.data)
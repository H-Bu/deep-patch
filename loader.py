import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class GN(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.transform(Image.fromarray(np.uint8(self.data[idx]))), self.label[idx]

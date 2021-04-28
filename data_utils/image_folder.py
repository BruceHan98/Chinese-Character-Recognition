import os
from PIL import Image
from torch.utils import data


class ImageFolder(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        target = 0
        return image, target

    def __len__(self):
        return len(self.image_paths)

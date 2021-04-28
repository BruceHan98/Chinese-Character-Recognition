import os
from PIL import Image
from torchvision import transforms
from torch.utils import data


class DataSet(data.Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.images = list()
        self.labels = list()

        if mode == 'train':
            root = config.train_data
        elif mode == 'test':
            root = config.test_data
        else:
            root = config.train_data

        categories = os.listdir(root)
        for category in categories:
            image_dir = os.listdir(os.path.join(root, category))
            image_paths = [os.path.join(root, category, str(image)) for image in image_dir]
            if mode == 'train':
                image_paths = image_paths[: int(0.8 * len(image_paths))]
            elif mode == 'test':
                image_paths = image_paths[:]
            else:
                image_paths = image_paths[int(0.8 * len(image_paths)):]
            self.images += image_paths
            self.labels += [category] * len(image_paths)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = int(self.labels[index][1:])

        Transform = transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.Pad(self.config.pad_size, fill=255),
            transforms.RandomRotation((-20, 20)),
            transforms.RandomAffine((-10, 10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8161, 0.8161, 0.8161], std=[0.2425, 0.2425, 0.2425])
        ])

        image = Transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


def data_loader(config, mode):
    dataset = DataSet(config, mode)
    loader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    return loader

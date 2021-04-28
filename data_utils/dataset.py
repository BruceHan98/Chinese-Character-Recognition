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
            transforms.RandomRotation((-20, 20)),
            transforms.RandomAffine((-10, 10)),
            transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8161, 0.8161, 0.8161], std=[0.2425, 0.2425, 0.2425])
        ])

        image = padding(image)
        image = Transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


def padding(image, min_size=96):  # PIL Image
    width, height = image.size
    size = max(min_size, width, height)
    new_im = Image.new('RGB', (size, size), (255, 255, 255))
    new_im.paste(image, (int((size - width) / 2), int((size - height) / 2)))
    return new_im


def data_loader(config, mode):
    dataset = DataSet(config, mode)
    loader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    return loader

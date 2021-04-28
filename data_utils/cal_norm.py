import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from image_folder import ImageFolder
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    )
    data_set = ImageFolder('../data/test', transform)
    data_loader = DataLoader(data_set, batch_size=64, num_workers=8, shuffle=False)

    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for images, targets in tqdm(data_loader):
        # scale image to be between 0 and 1
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


if __name__ == '__main__':
    main()
# tensor([0.8161, 0.8161, 0.8161]) tensor([0.2425, 0.2425, 0.2425])

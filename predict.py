import os
import torch
from network.network import Network
from utils.config import config
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np


def predict(image_dir):
    model = Network(config.class_num).eval()
    model.load_state_dict(torch.load(config.model_path, map_location='cpu'))
    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    model.to(device)
    torch.no_grad()

    test_images = list()
    for (root, dirs, filenames) in os.walk(image_dir):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if not filename.isdigit():
                continue
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                test_images.append(os.path.join(root, file))
    # print(test_images)
    test_images.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # 按数字排序

    dict_file = open('char_dict', 'rb')
    char_dict = pickle.load(dict_file)

    transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.GaussianBlur((3, 3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8161, 0.8161, 0.8161], std=[0.2425, 0.2425, 0.2425])
    ])

    for im in test_images:
        img = Image.open(im)
        img_ = transformer(img).unsqueeze(0)
        img_ = img_.to(device)
        outs = model(img_)

        top3 = torch.topk(outs, 3, dim=-1)[-1].tolist()
        top3 = np.array(top3).flatten()

        result = []
        for i in top3:
            char = list(char_dict.keys())[list(char_dict.values()).index(i)]
            result.append(char)
        print(result[0])


if __name__ == '__main__':
    predict('images/img')

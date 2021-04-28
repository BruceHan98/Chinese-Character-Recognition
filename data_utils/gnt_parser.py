import os
import time
import numpy as np
import struct
import pickle
from PIL import Image
from multiprocessing import Pool


def read_gnt(gnt_path):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    with open(gnt_path, 'rb') as f:
        for image, tagcode in one_file(f):
            yield image, tagcode


def test():
    count = 0
    for image, tagcode in read_gnt('gnt_test/001-f.gnt'):
        count += 1
        print(count)


def get_all_files(gnt_dir):
    file_list = list()
    for root, dirs, files in os.walk(gnt_dir):
        for f in files:
            if f.split('.')[-1] == "gnt":
                file_list.append(os.path.join(root, f))
    return file_list


def get_char_dict(gnt_dir):
    char_set = set()
    print("Generate character dictionary...")

    file_list = get_all_files(gnt_dir)
    for file in file_list:
        for _, tagcode in read_gnt(file):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312', 'ignore')
            char_set.add(tagcode_unicode)
    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))

    f = open('../char_dict', 'wb')
    pickle.dump(char_dict, f)
    f.close()
    print("done.")


def gen_dataset(gnt_path, index, save_dir):
    f = open('../char_dict', 'rb')
    char_dict = pickle.load(f)
    counter = 0
    for image, tagcode in read_gnt(gnt_path):
        image = Image.fromarray(image)
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312', 'ignore')
        category = str(char_dict[tagcode_unicode])

        if not os.path.exists(os.path.join(save_dir, category)):
            os.makedirs(os.path.join(save_dir, category))
        save_path = os.path.join(os.path.join(save_dir, category), str(index) + str(counter).zfill(4) + '.png')
        # print(save_path)

        image.convert('RGB').save(save_path)
        print(f"{counter} image saved", end='\r')
        counter += 1


def multi_process(gnt_dir, num_workers, save_dir):
    file_list = get_all_files(gnt_dir)
    p = Pool(num_workers)
    for i, f in enumerate(file_list):
        p.apply_async(gen_dataset, args=(f, i, save_dir))
    p.close()
    p.join()
    print("done.")


if __name__ == '__main__':
    gnt_train_dir = 'gnt_train'
    gnt_test_dir = 'gnt_test'
    train_data_dir = 'data/train_data'
    test_data_dir = 'data/test_data'
    num_workers = 8

    get_char_dict(gnt_train_dir)

    start_time = time.time()
    multi_process(gnt_train_dir, num_workers, train_data_dir)
    multi_process(gnt_test_dir, num_workers, test_data_dir)

    print("cost time: %.4f" % (time.time() - start_time))
    # test()

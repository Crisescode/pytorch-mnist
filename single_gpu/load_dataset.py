import os
from torch.utils.data.dataset import Dataset
import torchvision.datasets.mnist as mnist
from skimage import io
from torchvision import transforms
from PIL import Image


class LocalDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        with open(base_path) as fp:
            for line in fp.readlines():
                tmp = line.split(" ")
                self.data.append([tmp[0], tmp[1][7:8]])

        self.transformations = \
            transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

    def __getitem__(self, index):
        img = self.transformations(Image.open(self.data[index][0]))
        label = int(self.data[index][1])
        return img, label

    def __len__(self):
        return len(self.data)


class DataPreprocess(object):
    def __init__(self, root):
        self.root = root

    @property
    def get_train_set(self):
        train_set = (
            mnist.read_image_file(os.path.join(self.root, 'train-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(self.root, 'train-labels-idx1-ubyte')))
        return train_set

    @property
    def get_test_set(self):
        test_set = (
            mnist.read_image_file(os.path.join(self.root, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(os.path.join(self.root, 't10k-labels-idx1-ubyte')))
        return test_set

    def convert_to_img(self):
        f = open(self.root + 'train.txt', 'w')
        data_path = self.root + 'train/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(self.get_train_set[0], self.get_train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

        f = open(self.root + 'test.txt', 'w')
        data_path = self.root + 'test/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(self.get_test_set[0], self.get_test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

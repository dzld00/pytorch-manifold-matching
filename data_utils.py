from os import listdir
from os.path import join
from glob import glob
import csv
from collections import OrderedDict
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(size):
    transform = None
    transform = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    return transform


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, size):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = glob(dataset_dir + '/**/*.jpg', recursive=True)
        self.resize_transform = train_transform(size)

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB') # convert RGB if is color image
        real_image = self.resize_transform(image)
        #real_image = real_image * 2 - 1 # if use tanh
        return real_image

    def __len__(self):
        return len(self.image_filenames)

    
def generate_image(netG, dim, batch_size, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
    	noisev = noise 
    samples = netG(noisev)
    samples = samples.view(batch_size, 3, dim, dim) # 1 or 3
    #samples = (samples + 1) * 0.5 # if tanh
    return samples

def gen_rand_noise(batch_size, ):
    noise = torch.randn(batch_size, 128,1,1)
    return noise

def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename
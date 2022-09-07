import os

from matplotlib.cbook import index_of
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import h5py
import numpy as np
import torch
import torch.utils.data as data
#from data.__init__ import CustomDatasetDataLoader
class HfiveDataset(BaseDataset):
    """A dataset class for datasets that are in the *.h5 file type.

    It assumes that the directory '/path/to/data/train/train.h5' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """


        BaseDataset.__init__(self, opt)





        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_AB =opt.dataroot+"/"+opt.phase+"/train.h5"
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

                #folder=opt.dataroot+"/"+opt.phase+"/train.h5"
        hf = h5py.File(self.dir_AB, 'r')
        key = list(hf.keys())
        self.dic = [ np.array(hf.get(key[i])) for i in range(len(key))]


        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


        self.AB_paths = self.dir_AB

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        # print(data.DataLoader)

        # hf = h5py.File(self.dir_AB, 'r')
        # key = list(hf.keys())
        # dic = [ np.array(hf.get(key[i])) for i in range(len(key))]

        # read a image given a random integer index
        #AB_path = self.AB_paths[index]
        #AB = Image.open(AB_path).convert('RGB')

        AB =self.dic[index]  #AB =self.dic[index].mean(0) 
        AB=Image.fromarray(AB)
        w, h = AB.size
        #AB=torch.tensor(AB)
        # split AB image into A and B
        # h, w = AB.shape
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        #A=AB[0:w2,0:h]
        #A.reshape(1,1,w2,h)
        # A=Image.fromarray(A)
        #B=AB[w2:w,0:h]
        # B=Image.fromarray(A)
        #B.reshape(1,1,w2,h)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B,'A_paths': self.AB_paths, 'B_paths': self.AB_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dic)

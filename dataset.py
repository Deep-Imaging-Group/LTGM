import os
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import glob
import csv
import pydicom
# import ctlib_v2
import pickle

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, img_dir, anno_file=""):
        self.img_dir = img_dir
        self.anno_file = anno_file
        if self.anno_file == "":
            self.img_list = glob.glob(img_dir+"/*/*")
            self.tag = 0
        else:
            self.img_list = []
            self.tag = 1
            with open(self.anno_file,encoding='UTF-8') as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    dir_name = row[0]
                    for i in range(1,len(row)):
                        if row[i] == "0.0":
                            continue
                        file_name = "IM"+"%04d"%i+".dcm"
                        self.img_list.append(os.path.join(self.img_dir,dir_name,file_name))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([256,256])
             ])
    # need to overload
    def __len__(self):
        return len(self.img_list)

    # need to overload
    def __getitem__(self, idx):
        ds = pydicom.dcmread(self.img_list[idx]).pixel_array.astype("double")
        # print(ds.dtype)
        # ds = np.expand_dims(ds,axis=0)
        ds = ds / 3072
        ds = self.transform(ds)
        return ds,self.tag

class pre_MyDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = glob.glob(img_dir+"/*")

    # need to overload
    def __len__(self):
        return len(self.img_list)

    # need to overload
    def __getitem__(self, idx):
        f = open(self.img_list[idx],'rb')
        data = pickle.load(f)
        f.close()
        return data["sino"],data["back_img"],data["data"],data["label"]

class bad_Dataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = glob.glob(img_dir + "/*")
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([256,256])
             ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        f = open(self.img_list[idx],'rb')
        data = pickle.load(f)
        f.close()
        ds = self.transform(data["data"])
        return ds,data["target"]


class NDataset(Dataset):
    def __init__(self, img_dir, anno_file=""):
        self.img_dir = img_dir
        self.anno_file = anno_file
        if self.anno_file == "":
            self.img_list = glob.glob(img_dir+"/*/*")
            self.tag = 0
        else:
            self.img_list = []
            self.tag = 1
            with open(self.anno_file,encoding='UTF-8') as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                for row in f_csv:
                    dir_name = row[0]
                    for i in range(1,len(row)):
                        if row[i] == "0.0":
                            continue
                        file_name = "IM"+"%04d"%i+".dcm"
                        self.img_list.append(os.path.join(self.img_dir,dir_name,file_name))
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize([256,256])
             ])

    # need to overload
    def __len__(self):
        return len(self.img_list)

    # need to overload
    def __getitem__(self, idx):
        ds = pydicom.dcmread(self.img_list[idx]).pixel_array.astype("double")
        # print(ds.dtype)
        # ds = np.expand_dims(ds,axis=0)
        ds = ds / 3072
        ds = self.transform(ds)
        return ds,self.tag,self.img_list[idx]


# a = bad_Dataset('./BadNets/covid_train')
# print('1')

# f = open(self.img_list[idx],'rb')
# data = pickle.load(f)
# f.close()

# dataset = pre_MyDataset("F:\dataset\COVID-LDCT\Dataset-S1\COVID-S1-sino")
# dataset = pre_MyDataset("F:\dataset\COVID-LDCT\Dataset-S1\\Normal-S1-sino")
# dataloader = DataLoader(dataset=dataset, batch_size=2)
# #
# # # display
# for sino,back_img,data,label in dataloader:
#     # print(label,img)
#     # print(img)
#     plt.figure()
#     plt.imshow(data[0], cmap=plt.cm.bone)
#     plt.show()

# # Your Data Path
# img_dir = 'F:/dataset/COVID-LDCT/Dataset-S1/COVID-S1'
# anno_file = 'F:/dataset/COVID-LDCT/Dataset-S1/LDCT-SL-Labels-S1.csv'
# #
# dataset = MyDataset(img_dir, anno_file)
# dataloader = DataLoader(dataset=dataset, batch_size=2)
# #
# # # img_dir = 'Normal-S1'
# # # anno_file = ''
# #
# dataset = MyDataset(img_dir, anno_file)
# # dataloader = DataLoader(dataset=dataset, batch_size=2)
# #
# # # display
# for img,label in dataloader:
#     break
#     # print(label,img)
#     # print(img)
#     plt.figure()
#     plt.imshow(img[0], cmap=plt.cm.bone)
#     plt.show()

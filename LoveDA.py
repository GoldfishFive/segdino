import os
from math import ceil, floor

import numpy as np
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import utils
from PIL import Image

import torchvision.transforms.functional as F

class DataAugmentationDINO(object):
    def __init__(self):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first crop
        self.view_transform1 = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second crop
        self.view_transform2 = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            # utils.Solarization(0.2), # ImageOps.solarize()将高于阈值的所有像素值反转，
            normalize,
        ])

        # transformation for the local small crops
        # self.local_crops_number = local_crops_number
        # self.local_transfo = transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            # utils.GaussianBlur(p=0.5),
            # normalize,
        # ])

    def __call__(self, image, mode):
        crops = []
        if mode == "view1":
            crops.append(self.view_transform1(image))
        elif mode == "view2":
            crops.append(self.view_transform2(image))

        return crops

class LoveDAdataset(Dataset):
    def __init__(self, target_view = None,anchor_view = None,transform=None, datadir="/media/database/data4/wjy/datasets/Segmentation/loveda/Train/"):
        self.imglist = sorted(os.listdir(datadir))
        self.imglist = [os.path.join(datadir,i) for i in self.imglist]
        self.target_view = target_view
        self.anchor_view = anchor_view
        self.transform = transform


    def __getitem__(self, index):
        # print(self.imglist[index])
        x_path = self.imglist[index]
        img_mate = {"image_path":x_path,}
        img_x = Image.open(x_path)
        # img_x = cv2.imread(x_path,-1)
        imgs = []

        if self.transform is not None:
            # print(img_x.size,img_x.mode)
            # img_x.save("imagestest/origin_1024_1024.png")

            #  resized_crop
            i, j, h, w = self.target_view.get_params(img_x, self.target_view.scale, self.target_view.ratio)
            # print(i,j,h,w)

            # img_x = F.resized_crop(img_x, i, j, h, w, self.resizedcrop_transform.size, self.resizedcrop_transform.interpolation)
            # equ to below
            img = F.crop(img_x, i, j, h, w)
            W,H = img_mate['img_size'] = img.size # w1,h1
            mask = Image.new('L', img.size)
            # img.save('imagestest/croped_view1_before_transfom.png')
            # print(type(img),img.size)
            img_x = F.resize(img, self.target_view.size,self.target_view.interpolation)

            # second view is croped from first view for student
            i2, j2, h2, w2 = self.anchor_view.get_params(img, self.anchor_view.scale, self.anchor_view.ratio)
            img2 = F.crop(img, i2, j2, h2, w2)
            # print(i2, j2, h2, w2)
            # print(type(img2),img2.size)
            mask2 = Image.new('L', img2.size, 1)
            # mask2.save('imagestest/mask2.png')
            S,T = img_mate['img2_size'] = img2.size # w2,h2
            # img_mate['mask_size'] = mask.size # w2,h2
            img_mate['crop_pos'] =  (i2, j2) # x2,y2
            # print(S,T)
            # S = S if S + i2 <=W else W-i2
            # T = T if T + j2 <=H else H-j2
            # print(S,T)
            S_ = floor(S*32/W)
            T_ = floor(T*32/H)
            i_ = floor(i2*32/W)
            j_ = floor(j2*32/H)
            img_mate['crop32'] = (i_,j_,S_,T_) # X,Y,W,H of 32*32
            # print(i_,j_,S_,T_)  # 不是很准确
            # print(S_*T_)

            mask.paste(mask2,(i2, j2, i2+w2, j2+h2))
            # mask.paste(mask2,(j2, i2, j2+w2,i2+h2))
            # mask.save('imagestest/mask.png')

            # img2.save('imagestest/croped_view2_before_transfom.png')

            img_two = F.resize(img2, self.anchor_view.size,self.anchor_view.interpolation)

            view1 = self.transform(img_x, "view1")  #first view for teacher
            view2 = self.transform(img_two, "view2")  #second view for student
            # mask = mask.resize((512, 512),Image.NEAREST)
            resized_mask = mask.resize((64, 64),Image.NEAREST)
            resized_32_mask = mask.resize((32, 32),Image.NEAREST)
            resized_mask = np.array(resized_mask)
            resized_32_mask = np.array(resized_32_mask)

        return view1, view2, resized_mask, resized_32_mask, img_mate

    def __len__(self):
        return len(self.imglist)


if __name__ == "__main__":

    transform = DataAugmentationDINO()
    target_view = transforms.RandomResizedCrop(size=(512,512),scale=(0.45, 0.75))
    anchor_view = transforms.RandomResizedCrop(size=(224,224),scale=(0.25, 0.45))

    love = LoveDAdataset(target_view = target_view,anchor_view = anchor_view,transform=transform)

    print(love[1][0][0].size())
    print(love[1][1][0].size())
    print(love[1][2])

    # print(love[0][0][0][0].size)
    # print(love[0][0][0][1].size)
    # love[0][0][0][0].save('imagestest/croped_view1_after_transform.png')
    # love[0][0][0][1].save('imagestest/croped_view2_after_transform.png')
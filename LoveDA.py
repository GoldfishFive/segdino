import argparse
import os
from math import ceil, floor

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import utils
from PIL import Image, ImageFile

import torchvision.transforms.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
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

    def __call__(self, image, mode):
        if mode == "view1":
            return self.view_transform1(image)
        elif mode == "view2":
            return self.view_transform2(image)

class LoveDAdataset(Dataset):
    def __init__(self, target_view = None,anchor_view = None,transform=None,
                 datadir="", args=None):
        self.imglist = sorted(os.listdir(datadir))
        self.imglist = [os.path.join(datadir,i) for i in self.imglist]
        self.target_view = target_view
        self.anchor_view = anchor_view
        self.transform = transform
        self.args = args
        self.local_crops_number = args.local_crops_number


    def __getitem__(self, index):
        # print(self.imglist[index])
        x_path = self.imglist[index]
        img_mate = {"image_path":x_path,}
        img_x = Image.open(x_path)

        if self.transform is not None:
            # print(img_x.size,img_x.mode)
            # img_x.save("imagestest/origin_1024_1024.png")
            # img_x = F.resized_crop(img_x, i, j, h, w, self.resizedcrop_transform.size, self.resizedcrop_transform.interpolation)#  resized_crop
            # resized after crop
            i, j, h, w = self.target_view.get_params(img_x, self.target_view.scale, self.target_view.ratio)
            # print(i,j,h,w)
            img = F.crop(img_x, i, j, h, w)
            W,H = img_mate['img_size'] = img.size # w1,h1
            mask = Image.new('L', img.size)
            # img.save('imagestest/croped_view1_before_transfom.png')
            # print(type(img),img.size)
            img_x = F.resize(img, self.target_view.size,self.target_view.interpolation)
            view1 = self.transform(img_x, "view1")  #first view for teacher

            view2 = []
            resized_mask = []
            croped_from = []
            crop_pos = []
            # second view is croped from first view for student
            for idx in range(self.local_crops_number):
                i2, j2, h2, w2 = self.anchor_view.get_params(img, self.anchor_view.scale, self.anchor_view.ratio)
                img2 = F.crop(img, i2, j2, h2, w2)
                # print(i2, j2, h2, w2)
                # print(type(img2),img2.size)
                mask2 = Image.new('L', img2.size, 1)
                # mask2.save('imagestest/mask2.png')
                S,T = img2.size # w2,h2
                crop_pos.append((i2, j2))# x2,y2
                # print(idx, (i2, j2))
                '''
                print(S,T)
                S = S if S + i2 <=W else W-i2
                T = T if T + j2 <=H else H-j2
                print(S,T)
                '''
                patch_high_num = self.args.target_view_crops_size[0]/self.args.patch_size #32
                S_ = floor(S*patch_high_num/W)
                T_ = floor(T*patch_high_num/H)
                i_ = floor(i2*patch_high_num/W)
                j_ = floor(j2*patch_high_num/H)
                # print(i_,j_,S_,T_)  # 不是很准确
                croped_from.append((i_,j_,S_,T_))
                # print(S_*T_)
                mask.paste(mask2,(i2, j2, i2+w2, j2+h2))
                # mask.paste(mask2,(j2, i2, j2+w2,i2+h2))
                # mask.save('imagestest/mask.png')
                # img2.save('imagestest/croped_view2_before_transfom.png')

                img_two = F.resize(img2, self.anchor_view.size,self.anchor_view.interpolation)
                view2.append(self.transform(img_two, "view2"))  # second views for student
                resized_mask.append(np.array(mask.resize((64, 64),Image.NEAREST)))

            img_mate['crop_pos'] = crop_pos
            img_mate['croped_from'] = croped_from

        return view1, view2, resized_mask, img_mate

    def __len__(self):
        return len(self.imglist)



if __name__ == "__main__":
    from main_dino import get_args_parser
    parser = argparse.ArgumentParser('SegDINO', parents=[get_args_parser()])
    args = parser.parse_args()

    transform = DataAugmentationDINO()
    target_view = transforms.RandomResizedCrop(size=(512,512),scale=(0.45, 0.75))
    anchor_view = transforms.RandomResizedCrop(size=(224,224),scale=(0.25, 0.45))

    love = LoveDAdataset(target_view = target_view,anchor_view = anchor_view,transform=transform,datadir=args.data_path,args=args)
    view1, view2, target, img_mate = love[0]
    # print('view1:',view1)
    # print('view2:',view2)
    # print('target:',target)
    # print('img_mate:',img_mate)

    # print(love[0][0][0][0].size)
    # print(love[0][0][0][1].size)
    # love[0][0][0][0].save('imagestest/croped_view1_after_transform.png')
    # love[0][0][0][1].save('imagestest/croped_view2_after_transform.png')

    data_loader = torch.utils.data.DataLoader(
        love,
        batch_size=3,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    a = 1
    for it, (view1, view2, target, img_mate) in enumerate(data_loader):
        # print('view1:', view1)
        # print('view2:', view2)
        # print('target:', target)
        print('img_mate:', img_mate)

        for k,v in img_mate.items():
            print(k, v)

        print(len(view2))

        view1 = view1[0]
        view2_1  = view2[0]
        view2_2  = view2[1][0]
        target = target[0]
        print('view1.size():', view1.size())
        print('view2_2.size():', view2_2.size())
        print(view2_2)


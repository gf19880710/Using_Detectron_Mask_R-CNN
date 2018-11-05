# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image


class labelme2coco(object):
    def __init__(self, labelme_json=None, save_json_path='./new.json'):
        """
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.data_coco = None
        self.save_json()

    def data_transfer(self):
        """
        遍历所有labelme生成的json文件,进行COCO数据格式的转换
        :return: 无
        """
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label'].split('_')
                    if label[1] not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label[1])
                    points = shapes['points']
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        """
        获取图片基本信息
        :param data: json格式标注文件转换后的Python标准字典
        :param num: 索引
        :return: image字典, keys: height, width, id, file_name
        """
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        """
        获取图片种类信息
        :param label: 按照"_"分割之后的字符串列表
        :return: categorie字典, keys:　id, name, supercategory
        """
        categorie = dict()
        categorie['supercategory'] = label[0]
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = label[1]
        return categorie

    def annotation(self, points, label, num):
        """
        获取图片标注信息
        :param points: 标注区域点的集合
        :param label: 按照"_"分割之后的字符串列表
        :param num: 索引
        :return: annotation字典, keys: id, iamge_id, category_id, segmentation, bbox, iscrowd
        """
        annotation = dict()
        annotation['segmentation'] = [eval(str(list(np.asarray(points, dtype=np.float32).flatten())))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        annotation['area'] = self.poly_area(points)
        return annotation

    def getcatid(self, label):
        """
        获取标签对应的categoriy_id
        :param label: 按照"_"分割之后的字符串列表
        :return:
        """
        for categorie in self.categories:
            if label[1] == categorie['name']:
                return categorie['id']
        return -1

    def getbbox(self, points):
        """
        获取标注区域点所对应的bounding_box
        :param points: 标注区域点的集合
        :return: [x1, y1, w, h]
        """
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        """从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        """
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        """
        将组成多边形点的列表转换为掩膜
        :param img_shape: 图片的形状
        :param polygons: 组成多边形点的列表
        :return: 掩膜
        """
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        """
        将数据封装成为coco格式
        :return: data_coco, coco格式的数据
        """
        data_coco = dict()
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示

    def poly_area(self, points):
        """
        计算多边形面积
        :param points: 标注文件中points列表
        :return: 对应多边形面积
        """
        x_list = [coord[0] for coord in points]
        y_list = [coord[1] for coord in points]
        x = np.array(x_list)
        y = np.array(y_list)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


labelme_json = glob.glob('./*.json')
# labelme_json=['./1.json']

labelme2coco(labelme_json, './new.json')

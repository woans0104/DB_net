import numpy as np
import cv2

from .data_process import DataProcess
from concern.config import Configurable, State

import random
from PIL import Image
from torchvision.transforms.functional import resized_crop, crop
from torchvision.transforms import RandomResizedCrop, RandomCrop
from torchvision.transforms import InterpolationMode


# random crop algorithm similar to https://github.com/argman/EAST
class RandomCropData(DataProcess):
    size = State(default=(512, 512))
    max_tries = State(default=50)
    min_crop_side_ratio = State(default=0.1)
    require_original_image = State(default=False)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def crop_area(self, data, size=640):


        img = data
        img = Image.fromarray(img.astype('uint8'))
        short_side = min(img.size)
        top, left, height, width = RandomCrop.get_params(img, output_size=(short_side,short_side))
        img = resized_crop(img, top, left, height, width, size=(size, size),
                           interpolation=InterpolationMode.BICUBIC)

        img = np.array(img).astype('float32')

        return top, left, height, width, img


    # def process(self, data):
    #     img = data['image']
    #     ori_img = img
    #     ori_lines = data['polys']
    #
    #     all_care_polys = [line['points']
    #                       for line in data['polys'] if not line['ignore']]
    #     #crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
    #     crop_x, crop_y, crop_w, crop_h = self.crop_area(img)
    #     scale_w = self.size[0] / crop_w
    #     scale_h = self.size[1] / crop_h
    #     scale = min(scale_w, scale_h)
    #     h = int(crop_h * scale)
    #     w = int(crop_w * scale)
    #
    #     import ipdb;ipdb.set_trace()
    #
    #     padimg = np.zeros(
    #         (self.size[1], self.size[0], img.shape[2]), img.dtype)
    #     #
    #     # padimg[:h, :w] = cv2.resize(
    #     #     img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
    #
    #     img = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
    #
    #
    #     #img = padimg
    #
    #     lines = []
    #     for line in data['polys']:
    #         poly = ((np.array(line['points']) -
    #                  (crop_x, crop_y)) * scale).tolist()
    #         if not self.is_poly_outside_rect(poly, 0, 0, w, h):
    #             poly = self.fix_poly_outside_rect_oneside(poly, 0, 0, w, h)
    #             lines.append({**line, 'points': poly})
    #     data['polys'] = lines
    #
    #
    #
    #     #------------------------------------------------#
    #
    #     img_ = img.copy()
    #     for box in data['polys']:
    #         test_poly = box["points"]
    #
    #         cv2.polylines(
    #             img_, [np.reshape(np.int32(test_poly), (-1, 1, 2))], True, (0, 0, 255),10
    #         )
    #
    #     cv2.imwrite('testpoly_before.png', img_)
    #     # import ipdb;
    #     # ipdb.set_trace()
    #     #
    #     # try:
    #     #     cv2.imwrite('./testtest1/ori_{}'.format(data['filename']),ori_img)
    #     #     cv2.imwrite('./testtest1/crop_{}'.format(data['filename']), img)
    #     # except:
    #     #
    #     #     cv2.imwrite('./testtest1/ori_{}.png'.format(data['filename']),ori_img)
    #     #     cv2.imwrite('./testtest1/crop_{}.png'.format(data['filename']), img)
    #
    #     # ------------------------------------------------#
    #     if self.require_original_image:
    #         data['image'] = ori_img
    #     else:
    #         data['image'] = img
    #     data['lines'] = ori_lines
    #     data['scale_w'] = scale
    #     data['scale_h'] = scale
    #
    #     return data



    def process(self, data):

        img = data['image']
        ori_img = img
        ori_lines = data['polys']

        all_care_polys = [line['points']
                          for line in data['polys'] if not line['ignore']]
        #crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        crop_y, crop_x, crop_h, crop_w, img  = self.crop_area(img, self.size[0])


        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)


        lines = []
        for line in data['polys']:

            poly = ((np.array(line['points']) -
                     (crop_x, crop_y)) * scale).tolist()

            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                poly = self.fix_poly_outside_rect_oneside(poly, 0, 0, w, h)
                lines.append({**line, 'points': poly})
        data['polys'] = lines



        #------------------------------------------------#

        # img_ = img.copy()
        # for box in data['polys']:
        #     test_poly = box["points"]
        #
        #     cv2.polylines(
        #         img_, [np.reshape(np.int32(test_poly), (-1, 1, 2))], True, (0, 0, 255),10
        #     )
        #
        # cv2.imwrite('testpoly_before.png', img_)
        # import ipdb;
        # ipdb.set_trace()
        #
        # try:
        #     cv2.imwrite('./testtest1/ori_{}'.format(data['filename']),ori_img)
        #     cv2.imwrite('./testtest1/crop_{}'.format(data['filename']), img)
        # except:
        #
        #     cv2.imwrite('./testtest1/ori_{}.png'.format(data['filename']),ori_img)
        #     cv2.imwrite('./testtest1/crop_{}.png'.format(data['filename']), img)

        # ------------------------------------------------#
        if self.require_original_image:
            data['image'] = ori_img
        else:
            data['image'] = img
        data['lines'] = ori_lines
        data['scale_w'] = scale
        data['scale_h'] = scale

        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def fix_poly_outside_rect_oneside(self, poly, x, y, w, h):

        poly = np.array(poly)

        # 일부 영역이 외부에 있는 경우
        x_min = np.min(poly[:, 0])
        x_max = np.max(poly[:, 0])
        y_min = np.min(poly[:, 1])
        y_max = np.max(poly[:, 1])

        if x_max > x + w or y_max > y + h :
            if x_max > x + w:
                poly[:, 0] -= (x_max - (x + w))

            if y_max > y + h:
                poly[:, 1] -= (y_max - (y + h))

        #0807 code 수정-----------
        
        return poly.tolist()


    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        # 양쪽 다 밖에 있는 경우
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    # def crop_area(self, img, polys):
    #     h, w, _ = img.shape
    #     h_array = np.zeros(h, dtype=np.int32)
    #     w_array = np.zeros(w, dtype=np.int32)
    #     for points in polys:
    #         points = np.round(points, decimals=0).astype(np.int32)
    #         minx = np.min(points[:, 0])
    #         maxx = np.max(points[:, 0])
    #         w_array[minx:maxx] = 1
    #         miny = np.min(points[:, 1])
    #         maxy = np.max(points[:, 1])
    #         h_array[miny:maxy] = 1
    #     # ensure the cropped area not across a text
    #     h_axis = np.where(h_array == 0)[0]
    #     w_axis = np.where(w_array == 0)[0]
    #
    #     if len(h_axis) == 0 or len(w_axis) == 0:
    #         return 0, 0, w, h
    #
    #     h_regions = self.split_regions(h_axis)
    #     w_regions = self.split_regions(w_axis)
    #
    #     for i in range(self.max_tries):
    #         if len(w_regions) > 1:
    #             xmin, xmax = self.region_wise_random_select(w_regions, w)
    #         else:
    #             xmin, xmax = self.random_select(w_axis, w)
    #         if len(h_regions) > 1:
    #             ymin, ymax = self.region_wise_random_select(h_regions, h)
    #         else:
    #             ymin, ymax = self.random_select(h_axis, h)
    #
    #         if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
    #             # area too small
    #             continue
    #         num_poly_in_rect = 0
    #         for poly in polys:
    #             if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
    #                 num_poly_in_rect += 1
    #                 break
    #
    #         if num_poly_in_rect > 0:
    #             return xmin, ymin, xmax - xmin, ymax - ymin
    #
    #     return 0, 0, w, h

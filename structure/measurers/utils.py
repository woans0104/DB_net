

import os
import xml.etree.ElementTree as elemTree
import math
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPoint
from collections import Counter



def make_txt(result_str, save_folder, filename, dtype):
    # str문자열 받아서, 문자열을 txt형식 파일으로 만들어서 PATH에 저장하는 함수.

    if dtype == 'label':
        filename = filename + '_label'
    elif dtype == 'pred':
        filename = filename + '_pred'
    else:
        filename = filename + '_{}'.format(str(dtype))

    result_txt = os.path.join(
        save_folder, '{}.txt'.format(filename))

    text_file = open(result_txt, "w", encoding='utf8')
    text_file.write(result_str)
    text_file.close()

    return print('save_successful')


def easyocr_result_to_clEval(bounds):
    # To use PopEval metric, the output of EasyOCR should be reorganized into txt form.
    #
    # bounds 리스트를 받아서, 아래 예시와 같은 str 형식으로 바꿔주는 함수
    # ==============================================================
    # Example)
    # input :
    # [([[273, 33], [519, 33], [519, 73], [273, 73]],
    # '진료비 세부내역서',
    # 0.7085034251213074)]
    #
    # output :
    # 273,33,519,33,519,73,273,73,"진료비 세부내역서"
    # ===============================================================

    result = ''

    for i, bound_sample in enumerate(bounds):

        point_list = bound_sample['points']

        try:
            bound_word = bound_sample['ignore'].item()
        except:
            bound_word = ''


        if isinstance(bound_word,list):
            bound_word = ' OR '.join(str(i) for i in bound_word)


        point_list_flatten = [str(int(coordinate)) for point in point_list for
                              coordinate in point]
        point_str = ', '.join(point_list_flatten)


        if bound_word == 1:
            bound_word ='###'

        output = point_str +', ' +"\"{}\"".format(bound_word)
        result = result + output + '\n'

    return result
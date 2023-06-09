import abc
import numpy as np
import cv2
import Polygon as polygon3
#from Polygon.cPolygon import Polygon as polygon3
from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import Point

import math
from structure.measurers.clEval.arg_parser import PARAMS

MAX_FIDUCIAL_POINTS = 50


def get_midpoints(p1, p2):
    return (p1[0] + p2[0])/2, (p1[1] + p2[1])/2


def point_distance(p1, p2):
    distx = math.fabs(p1[0] - p2[0])
    disty = math.fabs(p1[1] - p2[1])
    return math.sqrt(distx * distx + disty * disty)


def point_angle(p1, p2):
    return (np.arctan2(p1[0]-p2[0], p1[1]-p2[1]) * 180 / np.pi + 360) % 360


def corner_continuous_check(index1, index2, index3, index4):
    pass


class Box(metaclass=abc.ABCMeta):
    def __init__(self, points, confidence, transcription):
        self.points = points
        self.confidence = confidence
        self.transcription = transcription
        self.is_dc = transcription == "###"

    @abc.abstractmethod
    def __and__(self, other) -> float:
        """Returns intersection between two objects"""
        pass

    @abc.abstractmethod
    def subtract(self, other):
        """polygon subtraction"""
        pass

    @abc.abstractmethod
    def center(self):
        pass

    @abc.abstractmethod
    def center_distance(self, other):
        """center distance between each box"""

    @abc.abstractmethod
    def diagonal_length(self) -> float:
        """Returns diagonal length for box-level"""
        pass

    @abc.abstractmethod
    def is_inside(self, x, y) -> bool:
        """Returns point (x, y) is inside polygon."""
        pass

    @abc.abstractmethod
    def make_polygon_obj(self):
        # TODO: docstring 좀 더 자세히 적기
        """Make polygon object to calculate for future"""
        pass

    @abc.abstractmethod
    def pseudo_character_center(self) -> list:
        """get character level boxes for TedEval pseudo center"""
        pass


class QUAD(Box):
    """Points should be x1,y1,...,x4,y4 (8 points) format"""
    def __init__(self, points, confidence=0.0, transcription=""):
        super(QUAD, self).__init__(points, confidence, transcription)
        self.polygon = self.make_polygon_obj()
        if self.is_dc:
            self.transcription = "#" * self.pseudo_transcription_length()

    def __and__(self, other) -> float:
        """Get intersection between two area"""
        poly_intersect = self.polygon & other.polygon
        if len(poly_intersect) == 0:
            return 0.0
        return poly_intersect.area()

    def subtract(self, other):
        self.polygon = self.polygon - other.polygon

    def center(self):
        return self.polygon.center()

    def center_distance(self, other):
        return point_distance(self.center(), other.center())

    def area(self):
        return self.polygon.area()

    def __or__(self, other):
        return self.polygon.area() + other.polygon.area() - (self & other)

    def make_polygon_obj(self):
        point_matrix = np.empty((4, 2), np.int32)
        point_matrix[0][0] = int(self.points[0])
        point_matrix[0][1] = int(self.points[1])
        point_matrix[1][0] = int(self.points[2])
        point_matrix[1][1] = int(self.points[3])
        point_matrix[2][0] = int(self.points[4])
        point_matrix[2][1] = int(self.points[5])
        point_matrix[3][0] = int(self.points[6])
        point_matrix[3][1] = int(self.points[7])

        return polygon3.Polygon(point_matrix)

    def aspect_ratio(self):
        top_side = point_distance((self.points[0], self.points[1]), (self.points[2], self.points[3]))
        right_side = point_distance((self.points[2], self.points[3]), (self.points[4], self.points[5]))
        bottom_side = point_distance((self.points[4], self.points[5]), (self.points[6], self.points[7]))
        left_side = point_distance((self.points[6], self.points[7]), (self.points[0], self.points[1]))
        avg_hor = (top_side + bottom_side) / 2
        avg_ver = (right_side + left_side) / 2

        return (avg_ver + 1e-5) / (avg_hor + 1e-5)

    def pseudo_transcription_length(self):
        return min(round(0.5+(max(self.aspect_ratio(), 1/self.aspect_ratio()))), 10)

    def pseudo_character_center(self):
        chars = list()
        length = len(self.transcription)
        aspect_ratio = self.aspect_ratio()

        if length == 0:
            return chars

        if aspect_ratio < PARAMS.VERTICAL_ASPECT_RATIO_THRES:
            left_top = self.points[0], self.points[1]
            right_top = self.points[2], self.points[3]
            right_bottom = self.points[4], self.points[5]
            left_bottom = self.points[6], self.points[7]
        else:
            left_top = self.points[6], self.points[7]
            right_top = self.points[0], self.points[1]
            right_bottom = self.points[2], self.points[3]
            left_bottom = self.points[4], self.points[5]

        p1 = get_midpoints(left_top, left_bottom)
        p2 = get_midpoints(right_top, right_bottom)

        unit_x = (p2[0] - p1[0]) / length
        unit_y = (p2[1] - p1[1]) / length

        for i in range(length):
            x = p1[0] + unit_x / 2 + unit_x * i
            y = p1[1] + unit_y / 2 + unit_y * i
            chars.append((x, y))
        return chars

    def diagonal_length(self) -> float:
        left_top = self.points[0], self.points[1]
        right_top = self.points[2], self.points[3]
        right_bottom = self.points[4], self.points[5]
        left_bottom = self.points[6], self.points[7]
        diag1 = point_distance(left_top, right_bottom)
        diag2 = point_distance(right_top, left_bottom)
        return (diag1 + diag2) / 2
    
    def is_inside(self, x, y) -> bool:
        return self.polygon.isInside(x, y)


class POLY(Box):
    """Points should be x1,y1,...,xn,yn (2*n points) format"""
    def __init__(self, points, confidence=0.0, transcription=""):
        super(POLY, self).__init__(points, confidence, transcription)
        self.num_points = len(self.points) // 2
        self.polygon = self.make_polygon_obj()
        self._aspect_ratio = self.make_aspect_ratio()
        if self.is_dc:
            self.transcription = "#" * self.pseudo_transcription_length()

    def __and__(self, other):
        """Get intersection between two area"""
        poly_intersect = self.polygon.intersection(other.polygon)
        return poly_intersect.area

    def subtract(self, other):
        """get substraction"""
        self.polygon = self.polygon.difference(self.polygon.intersection(other.polygon))

    def __or__(self, other):
        return 1.0

    def area(self):
        return self.polygon.area

    def center(self):
        return self.polygon.centroid.coords[0]

    def center_distance(self, other):
        try:
            return point_distance(self.center(), other.center())
        except:
            return 0.0001

    def diagonal_length(self):
        left_top = self.points[0], self.points[1]
        right_top = self.points[self.num_points-2], self.points[self.num_points-1]
        right_bottom = self.points[self.num_points], self.points[self.num_points+1]
        left_bottom = self.points[self.num_points*2-2], self.points[self.num_points*2-1]

        diag1 = point_distance(left_top, right_bottom)
        diag2 = point_distance(right_top, left_bottom)

        return (diag1 + diag2) / 2
    
    def is_inside(self, x, y) -> bool:
        return self.polygon.contains(Point(x, y))

    def check_corner_points_are_continuous(self, lt, rt, rb, lb):
        count = 0
        while lt != rt:
            lt = (lt+1) % self.num_points
            count += 1

        while rb != lb:
            rb = (rb+1) % self.num_points
            count += 1

        return True

    def get_four_max_distance_from_center(self):
        center_x, center_y = self.center()
        distance_from_center = list()
        point_x = self.points[0::2]
        point_y = self.points[1::2]

        for px, py in zip(point_x, point_y):
            distance_from_center.append(point_distance((center_x, center_y), (px, py)))

        distance_idx_max_order = np.argsort(distance_from_center)[::-1]
        return distance_idx_max_order[:4]

    def make_polygon_obj(self):
        point_x = self.points[0::2]
        point_y = self.points[1::2]
        # in TotalText dataset, there there are under 4 points annotation for Polygon shape
        # so, we have to deal with it

        # if points are given 3, fill last quad points with left bottom coordinates
        if len(point_x) == len(point_y) == 3:
            point_x.append(point_x[0])
            point_y.append(point_y[2])
            self.points.append(point_x[0])
            self.points.append(point_y[2])
            self.num_points = len(self.points) // 2
        # if points are given 2, copy value 2 times
        elif len(point_x) == len(point_y) == 2:
            point_x *= 2
            point_y *= 2
            self.points.append(point_x[1])
            self.points.append(point_y[0])
            self.points.append(point_x[0])
            self.points.append(point_y[1])
            self.num_points = len(self.points) // 2
        # if points are given 1, copy value 4 times
        elif len(point_x) == len(point_y) == 1:
            point_x *= 4
            point_y *= 4
            for _ in range(3):
                self.points.append(point_x[0])
                self.points.append(point_x[0])
            self.num_points = len(self.points) // 2
        return shapely_poly(np.stack([point_x, point_y], axis=1)).buffer(0)

    def aspect_ratio(self):
        return self._aspect_ratio

    def pseudo_transcription_length(self):
        return min(round(0.5+ (max(self._aspect_ratio, 1/self._aspect_ratio))), 10)

    def make_aspect_ratio(self):
        np.array(np.reshape(self.points, [-1, 2]))
        rect = cv2.minAreaRect(np.array(np.reshape(self.points, [-1, 2]), dtype=np.float32))
        width = rect[1][0]
        height = rect[1][1]

        width += 1e-6
        height += 1e-6

        return min(10, height/width) + 1e+5

    def pseudo_character_center(self):
        chars = list()
        length = len(self.transcription)

        # Prepare polygon line estimation with interpolation
        point_x = self.points[0::2]
        point_y = self.points[1::2]
        points_x_top = point_x[:self.num_points//2]
        points_x_bottom = point_x[self.num_points//2:]
        points_y_top = point_y[:self.num_points//2]
        points_y_bottom = point_y[self.num_points//2:]

        # reverse bottom point order from left to right
        points_x_bottom = points_x_bottom[::-1]
        points_y_bottom = points_y_bottom[::-1]

        num_interpolation_section = (self.num_points//2) - 1
        num_points_to_interpolate = length

        new_point_x_top, new_point_x_bottom = list(), list()
        new_point_y_top, new_point_y_bottom = list(), list()

        for sec_idx in range(num_interpolation_section):
            start_x_top, end_x_top = points_x_top[sec_idx], points_x_top[sec_idx+1]
            start_y_top, end_y_top = points_y_top[sec_idx], points_y_top[sec_idx+1]
            start_x_bottom, end_x_bottom = points_x_bottom[sec_idx], points_x_bottom[sec_idx + 1]
            start_y_bottom, end_y_bottom = points_y_bottom[sec_idx], points_y_bottom[sec_idx + 1]

            diff_x_top = (end_x_top - start_x_top) / num_points_to_interpolate
            diff_y_top = (end_y_top - start_y_top) / num_points_to_interpolate
            diff_x_bottom = (end_x_bottom - start_x_bottom) / num_points_to_interpolate
            diff_y_bottom = (end_y_bottom - start_y_bottom) / num_points_to_interpolate

            new_point_x_top.append(start_x_top)
            new_point_x_bottom.append(start_x_bottom)
            new_point_y_top.append(start_y_top)
            new_point_y_bottom.append(start_y_bottom)

            for num_pt in range(1, num_points_to_interpolate):
                new_point_x_top.append(int(start_x_top + diff_x_top * num_pt))
                new_point_x_bottom.append(int(start_x_bottom + diff_x_bottom * num_pt))
                new_point_y_top.append(int(start_y_top + diff_y_top * num_pt))
                new_point_y_bottom.append(int(start_y_bottom + diff_y_bottom * num_pt))
        new_point_x_top.append(points_x_top[-1])
        new_point_y_top.append(points_y_top[-1])
        new_point_x_bottom.append(points_x_bottom[-1])
        new_point_y_bottom.append(points_y_bottom[-1])

        len_section_for_single_char = (len(new_point_x_top)-1) / len(self.transcription)
        print(self.num_points)
        print(len(self.transcription))

        for c in range(len(self.transcription)):
            print(len(new_point_x_top), c, len_section_for_single_char)
            center_x = (new_point_x_top[int(c*len_section_for_single_char)] +
                        new_point_x_top[int((c+1)*len_section_for_single_char)] +
                        new_point_x_bottom[int(c*len_section_for_single_char)] +
                        new_point_x_bottom[int((c+1)*len_section_for_single_char)]) / 4

            center_y = (new_point_y_top[int(c*len_section_for_single_char)] +
                        new_point_y_top[int((c+1)*len_section_for_single_char)] +
                        new_point_y_bottom[int(c*len_section_for_single_char)] +
                        new_point_y_bottom[int((c+1)*len_section_for_single_char)]) / 4

            chars.append((center_x, center_y))
        return chars

    # def pseudo_character_center(self):
    #     chars = list()

    #     # TODO: write comment more fancier(?)
    #     # Prepare polygon line estimation with interpolation

    #     point_x = self.points[0::2]
    #     point_y = self.points[1::2]
    #     points_x_top = point_x[:self.num_points//2]
    #     points_x_bottom = point_x[self.num_points//2:]
    #     points_y_top = point_y[:self.num_points//2]
    #     points_y_bottom = point_y[self.num_points//2:]

    #     points_x_bottom = points_x_bottom[::-1]
    #     points_y_bottom = points_y_bottom[::-1]
    #     # print(points_x_top, points_x_bottom, points_y_top, points_y_bottom)

    #     num_interpolation_section = (self.num_points//2) - 1
    #     num_points_to_interpolate = MAX_FIDUCIAL_POINTS // num_interpolation_section

    #     # TODO: using computer vision?
    #     new_point_x_top, new_point_x_bottom = list(), list()
    #     new_point_y_top, new_point_y_bottom = list(), list()

    #     for sec_idx in range(num_interpolation_section):
    #         start_x_top, end_x_top = points_x_top[sec_idx], points_x_top[sec_idx+1]
    #         start_y_top, end_y_top = points_y_top[sec_idx], points_y_top[sec_idx+1]
    #         start_x_bottom, end_x_bottom = points_x_bottom[sec_idx], points_x_bottom[sec_idx + 1]
    #         start_y_bottom, end_y_bottom = points_y_bottom[sec_idx], points_y_bottom[sec_idx + 1]

    #         diff_x_top = (end_x_top - start_x_top) / num_points_to_interpolate
    #         diff_y_top = (end_y_top - start_y_top) / num_points_to_interpolate
    #         diff_x_bottom = (end_x_bottom - start_x_bottom) / num_points_to_interpolate
    #         diff_y_bottom = (end_y_bottom - start_y_bottom) / num_points_to_interpolate

    #         new_point_x_top.append(start_x_top)
    #         new_point_x_bottom.append(start_x_bottom)
    #         new_point_y_top.append(start_y_top)
    #         new_point_y_bottom.append(start_y_bottom)

    #         for num_pt in range(1, num_points_to_interpolate):
    #             new_point_x_top.append(int(start_x_top + diff_x_top * num_pt))
    #             new_point_x_bottom.append(int(start_x_bottom + diff_x_bottom * num_pt))
    #             new_point_y_top.append(int(start_y_top + diff_y_top * num_pt))
    #             new_point_y_bottom.append(int(start_y_bottom + diff_y_bottom * num_pt))

    #     len_section_for_single_char = len(new_point_x_top) / len(self.transcription)

    #     for c in range(len(self.transcription)):
    #         center_x = (new_point_x_top[int(c*len_section_for_single_char)] +
    #                     new_point_x_top[min(int((c+1)*len_section_for_single_char), len(new_point_x_top)-1)] +
    #                     new_point_x_bottom[int(c*len_section_for_single_char)] +
    #                     new_point_x_bottom[min(int((c+1)*len_section_for_single_char), len(new_point_x_top)-1)]) / 4

    #         center_y = (new_point_y_top[int(c*len_section_for_single_char)] +
    #                     new_point_y_top[min(int((c+1)*len_section_for_single_char), len(new_point_x_top)-1)] +
    #                     new_point_y_bottom[int(c*len_section_for_single_char)] +
    #                     new_point_y_bottom[min(int((c+1)*len_section_for_single_char), len(new_point_x_top)-1)]) / 4

    #         chars.append((center_x, center_y))
    #     return chars

    # def pseudo_character_center_using_curve_fitting(self):
    #     if self.num_points == 4:
    #         return self.pseudo_character_center_prev()
    #
    #     chars = list()
    #
    #     point_x = self.points[0::2]
    #     point_y = self.points[1::2]
    #
    #     center_x, center_y = self.center()
    #
    #     # calculate distance between each point & center
    #     distance = []
    #     for px, py in zip(point_x, point_y):
    #         distance.append(point_distance((center_x, center_y), (px, py)))
    #
    #     # filter 4 points with maximum distance (to find each corner points)
    #     distance_max_idx_order = np.argsort(distance)[::-1][:4]
    #
    #     # distinguish left/right point by x coordinate difference with center point
    #     left_candidate_indices = []
    #     right_candidate_indices = []
    #
    #     for idx in distance_max_idx_order:
    #         if point_x[idx] < center_x:
    #             left_candidate_indices.append(idx)
    #         else:
    #             right_candidate_indices.append(idx)
    #
    #     # there should be 2: 2 points for each left/right location
    #     print(self.transcription, left_candidate_indices, right_candidate_indices)
    #
    #     # case
    #     # find minimum distance point with single left/right points, and append it to left/right candidate
    #     if len(left_candidate_indices) == 0 or len(right_candidate_indices) == 0:
    #         left_candidate_indices = [0, self.num_points-1]
    #         right_candidate_indices = [self.num_points//2-1, self.num_points//2]
    #
    #     if not len(left_candidate_indices) == len(right_candidate_indices):
    #         distance_between_corner = list()
    #         if len(left_candidate_indices) == 1:
    #             for right_idx in right_candidate_indices:
    #                 distance_between_corner.append(point_distance((point_x[right_idx], point_y[right_idx]),
    #                                                               (point_x[left_candidate_indices[0]],
    #                                                                point_y[left_candidate_indices[0]])))
    #         elif len(right_candidate_indices) == 1:
    #             for left_idx in left_candidate_indices:
    #                 distance_between_corner.append(point_distance((point_x[left_idx], point_y[left_idx]),
    #                                                               (point_x[right_candidate_indices[0]],
    #                                                                point_y[right_candidate_indices[0]])))
    #
    #         min_distance_point_idx = np.argsort(distance_between_corner)[0]
    #         if len(left_candidate_indices) == 1:
    #             left_candidate_indices.append(right_candidate_indices[min_distance_point_idx])
    #             right_candidate_indices.pop(min_distance_point_idx)
    #         else:
    #             right_candidate_indices.append(left_candidate_indices[min_distance_point_idx])
    #             left_candidate_indices.pop(min_distance_point_idx)
    #
    #     # distinguish top / bottom points by y coordinate difference compared with each left/right points
    #     if point_angle((point_x[left_candidate_indices[0]], point_y[left_candidate_indices[0]]), (center_x, center_y)) <\
    #        point_angle((point_x[left_candidate_indices[1]], point_y[left_candidate_indices[1]]), (center_x, center_y)):
    #         left_top_idx = left_candidate_indices[0]
    #         left_bottom_idx = left_candidate_indices[1]
    #     else:
    #         left_top_idx = left_candidate_indices[1]
    #         left_bottom_idx = left_candidate_indices[0]
    #
    #     if point_angle((point_x[right_candidate_indices[0]], point_y[right_candidate_indices[0]]), (center_x, center_y)) > \
    #        point_angle((point_x[right_candidate_indices[1]], point_y[right_candidate_indices[1]]), (center_x, center_y)):
    #         right_top_idx = right_candidate_indices[0]
    #         right_bottom_idx = right_candidate_indices[1]
    #     else:
    #         right_top_idx = right_candidate_indices[1]
    #         right_bottom_idx = right_candidate_indices[0]
    #
    #     # if point_y[left_candidate_indices[0]] < point_y[left_candidate_indices[1]]:
    #     #     left_top_idx = left_candidate_indices[0]
    #     #     left_bottom_idx = left_candidate_indices[1]
    #     # else:
    #     #     left_top_idx = left_candidate_indices[1]
    #     #     left_bottom_idx = left_candidate_indices[0]
    #     #
    #     # if point_y[right_candidate_indices[0]] < point_y[right_candidate_indices[1]]:
    #     #     right_top_idx = right_candidate_indices[0]
    #     #     right_bottom_idx = right_candidate_indices[1]
    #     # else:
    #     #     right_top_idx = right_candidate_indices[1]
    #     #     right_bottom_idx = right_candidate_indices[0]
    #
    #     top_x, top_y = list(), list()
    #     bottom_x, bottom_y = list(), list()
    #     top_start = left_top_idx
    #     top_end = right_top_idx
    #     bottom_start = right_bottom_idx
    #     bottom_end = left_bottom_idx
    #
    #     self.check_corner_points_are_continuous(left_top_idx, right_top_idx, right_bottom_idx, left_bottom_idx)
    #
    #     while top_start != top_end:
    #         top_x.append(point_x[top_start])
    #         top_y.append(point_y[top_start])
    #         top_start = (top_start+1) % self.num_points
    #     top_x.append(point_x[top_end])
    #     top_y.append(point_y[top_end])
    #
    #     while bottom_start != bottom_end:
    #         bottom_x.append(point_x[bottom_start])
    #         bottom_y.append(point_y[bottom_start])
    #         bottom_start = (bottom_start+1) % self.num_points
    #
    #     bottom_x.append(point_x[bottom_end])
    #     bottom_y.append(point_y[bottom_end])
    #
    #     # reverse bottom points to make it left -> right order
    #     bottom_x = bottom_x[::-1]
    #     bottom_y = bottom_y[::-1]
    #
    #     # pseudo interpolation for more accurate polynomial estimation
    #     new_point_x_top, new_point_x_bottom = list(), list()
    #     new_point_y_top, new_point_y_bottom = list(), list()
    #
    #     NUM_POINTS_TO_FILL = 3
    #
    #     new_point_x_top.append(top_x[0])
    #     new_point_y_top.append(top_y[0])
    #     for sec_idx in range(len(top_x)-1):
    #         start_top_x, end_top_x = top_x[sec_idx], top_x[sec_idx+1]
    #         start_top_y, end_top_y = top_y[sec_idx], top_y[sec_idx+1]
    #
    #         diff_x_top = (end_top_x - start_top_x) / NUM_POINTS_TO_FILL
    #         diff_y_top = (end_top_y - start_top_y) / NUM_POINTS_TO_FILL
    #
    #         new_point_x_top.append(start_top_x)
    #         new_point_y_top.append(start_top_y)
    #
    #         for num_pt in range(1, NUM_POINTS_TO_FILL+1):
    #             new_point_x_top.append(int(start_top_x + diff_x_top * num_pt))
    #             new_point_y_top.append(int(start_top_y + diff_y_top * num_pt))
    #
    #     new_point_x_bottom.append(bottom_x[0])
    #     new_point_y_bottom.append(bottom_y[0])
    #     for sec_idx in range(len(bottom_x) - 1):
    #         start_bottom_x, end_bottom_x = bottom_x[sec_idx], bottom_x[sec_idx + 1]
    #         start_bottom_y, end_bottom_y = bottom_y[sec_idx], bottom_y[sec_idx + 1]
    #
    #         diff_x_top = (end_bottom_x - start_bottom_x) / NUM_POINTS_TO_FILL
    #         diff_y_top = (end_bottom_y - start_bottom_y) / NUM_POINTS_TO_FILL
    #
    #         new_point_x_bottom.append(start_bottom_x)
    #         new_point_y_bottom.append(start_bottom_y)
    #
    #         for num_pt in range(1, NUM_POINTS_TO_FILL + 1):
    #             new_point_x_bottom.append(int(start_bottom_x + diff_x_top * num_pt))
    #             new_point_y_bottom.append(int(start_bottom_y + diff_y_top * num_pt))
    #
    #     # do polynomial curve fitting
    #     top_curve = np.poly1d(np.polyfit(np.array(new_point_x_top), np.array(new_point_y_top), 2))
    #     bottom_curve = np.poly1d(np.polyfit(np.array(new_point_x_bottom), np.array(new_point_y_bottom), 2))
    #
    #     # # do polynomial curve fitting
    #     # top_curve = np.poly1d(np.polyfit(np.array(top_x), np.array(top_y), 2))
    #     # bottom_curve = np.poly1d(np.polyfit(np.array(bottom_x), np.array(bottom_y), 2))
    #
    #     # point interpolation
    #     interpolated_top_x = np.linspace(top_x[0], top_x[-1], len(self.transcription)+1)
    #     interpolated_bottom_x = np.linspace(bottom_x[0], bottom_x[-1], len(self.transcription)+1)
    #
    #     # estimate middle point from top/bottom polynomial
    #     interpolated_top_y = top_curve(interpolated_top_x)
    #     interpolated_bottom_y = bottom_curve(interpolated_bottom_x)
    #     middle_curve_point_x = (interpolated_top_x + interpolated_bottom_x) / 2
    #     middle_curve_point_y = (interpolated_top_y + interpolated_bottom_y) / 2
    #
    #     # fill character center list from middle curve information
    #     for c in range(len(self.transcription)):
    #         chars.append(((middle_curve_point_x[c] + middle_curve_point_x[c+1])/2,
    #                       (middle_curve_point_y[c] + middle_curve_point_y[c+1])/2))
    #
    #     return chars


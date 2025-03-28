import sys
import numpy as np

def check_dict(d, name):
    if len(d) < 1:
        print(f'{name} is empty')


def get_contours_from_annotations(target_structure, hand_annotations, densify=0):
    num_annotations = len(hand_annotations)
    contours_for_structure = {}
    for i in range(num_annotations):
        structure = hand_annotations['name'][i]
        side = hand_annotations['side'][i]
        section = hand_annotations['section'][i]
        first_section = 0
        last_section = 0
        if side == 'R' or side == 'L':
            structure = structure + '_' + side
        if structure == target_structure:
            vertices = hand_annotations['vertices'][i]
            for _ in range(densify):
                vertices = get_dense_coordinates(vertices)
            contours_for_structure[section] = vertices
    try:
        first_section = np.min(list(contours_for_structure.keys()))
        last_section = np.max(list(contours_for_structure.keys()))
    except:
        print(f'Could not get first and last section of {target_structure}')
        pass
    return contours_for_structure, first_section, last_section


def get_dense_coordinates(coor_list):
    dense_coor_list = []
    for i in range(len(coor_list) - 1):
        x, y = coor_list[i]
        x_next, y_next = coor_list[i + 1]
        x_mid = (x + x_next) / 2
        y_mid = (y + y_next) / 2
        dense_coor_list.append([x, y])
        dense_coor_list.append([x_mid, y_mid])
        if i == len(coor_list) - 2:
            dense_coor_list.append([x_next, y_next])
            x, y = coor_list[0]
            x_mid = (x + x_next) / 2
            y_mid = (y + y_next) / 2
            dense_coor_list.append([x_mid, y_mid])
    return dense_coor_list
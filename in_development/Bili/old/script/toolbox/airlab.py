"""AirLab-related utilities"""
import json

import numpy as np

class AirLabAffineTransform:
    def __init__(
        self, mov_spacing, fix_spacing, box_size,
        affine_matrix, affine_matrix_inv,
        t, phi, scale, shear
    ):
        self.mov_spacing = np.array(mov_spacing)
        self.fix_spacing = np.array(fix_spacing)
        self.box_size = np.array(box_size)
        self.affine_matrix = np.array(affine_matrix)
        self.affine_matrix_inv = np.array(affine_matrix_inv)
        self.t = tuple(t)
        self.phi = tuple(phi)
        self.scale = tuple(scale)
        self.shear = tuple(shear)

    def forward_point(self, coord):
        coord = np.array(coord)
        coord /= self.mov_spacing
        coord = coord / self.box_size * 2 - 1
        coord = np.append(coord, 1) @ self.affine_matrix
        coord = (coord + 1) / 2 * self.box_size
        coord *= self.fix_spacing
        return coord

    def init_al_transform(self, al_transform):
        al_transform.set_parameters(
            t=self.t, phi=self.phi, scale=self.scale, shear=self.shear
        )

def dump_al_affine_transform(mov_img, fix_img, al_transform, dump_file):
    affine_matrix = al_transform.transformation_matrix.t().detach().numpy()

    a = affine_matrix[:3,:]
    t = affine_matrix[3,:]
    a_inv = np.linalg.inv(a)
    t_inv = - t @ a_inv
    affine_matrix_inv = np.vstack((a_inv, t_inv))

    transform_param = {}
    transform_param['mov_spacing'] = mov_img.spacing.tolist()
    transform_param['fix_spacing'] = fix_img.spacing.tolist()
    transform_param['box_size'] = list(fix_img.size)
    transform_param['affine_matrix'] = affine_matrix_inv.tolist()
    transform_param['affine_matrix_inv'] = affine_matrix.tolist()
    transform_param['t'] = [
        float(al_transform._t_x),
        float(al_transform._t_y),
        float(al_transform._t_z)
    ]
    transform_param['phi'] = [
        float(al_transform._phi_z),
        float(al_transform._phi_x),
        float(al_transform._phi_y),
    ]
    transform_param['scale'] = [
        float(al_transform._scale_x),
        float(al_transform._scale_y),
        float(al_transform._scale_z)
    ]
    transform_param['shear'] = [
        float(al_transform._shear_y_x),
        float(al_transform._shear_x_y),
        float(al_transform._shear_z_x),
        float(al_transform._shear_z_y),
        float(al_transform._shear_x_z),
        float(al_transform._shear_y_z),
    ]
    with open(dump_file, 'w') as f:
        json.dump(transform_param, f, indent=4)

def load_al_affine_transform(dump_file):
    with open(dump_file, 'r') as f:
        transform_param = json.load(f)
    transform = AirLabAffineTransform(
        transform_param['mov_spacing'],
        transform_param['fix_spacing'],
        transform_param['box_size'],
        transform_param['affine_matrix'],
        transform_param['affine_matrix_inv'],
        transform_param['t'],
        transform_param['phi'],
        transform_param['scale'],
        transform_param['shear']
    )
    return transform

import argparse
import numpy as np
import math


def rigid_transform_2d(angle_deg: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """
    Return a 3x3 homogeneous rigid transform in 2D for a rotation by `angle_deg`
    (degrees) about the origin and translation (tx, ty).
    """
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    # 3x3 homogeneous matrix
    T = np.array([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0,  1]
    ], dtype=float)
    return T

def _rot3_about_axis(angle_rad: float, axis: str) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c,-s],
                      [0, s, c]], dtype=float)
    elif axis == 'y':
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=float)
    elif axis == 'z':
        R = np.array([[c,-s, 0],
                      [s, c, 0],
                      [0, 0, 1]], dtype=float)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    return R

def rigid_transform_3d(degrees: float, axis: str = 'z', tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """
    Return a 4x4 homogeneous rigid transform in 3D: rotation by `angle_deg` (degrees)
    about `axis` ('x', 'y' or 'z') and translation (tx,ty,tz).
    """
    theta = math.radians(degrees)
    R = _rot3_about_axis(theta, axis)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    print(T)



def create_quarternion(degrees, rotateby="x"):

    theta = math.radians(degrees)

    #theta = np.pi * 3/4


    if rotateby == "x":
        vec1 = np.array([1, 0, 0], dtype=float)
        vec3 = np.array([0, np.cos(theta), np.sin(theta)], dtype=float)
        vec4 = np.array([0, -np.sin(theta), np.cos(theta)], dtype=float)
    elif rotateby == "y":
        vec1 = np.array([np.cos(theta), 0, -np.sin(theta)], dtype=float)
        vec3 = np.array([1, 0, 0], dtype=float)
        vec4 = np.array([np.sin(theta), 1, np.cos(theta)], dtype=float)
    elif rotateby == "z":
        vec1 = np.array([np.cos(theta), -np.sin(theta), 0], dtype=float)
        vec3 = np.array([np.sin(theta), np.cos(theta), 1], dtype=float)
        vec4 = np.array([1, 0, 0], dtype=float)
    else:
        vec1 = np.array([1, 0, 0], dtype=float)
        vec3 = np.array([0, 1, 0], dtype=float)
        vec4 = np.array([0, 0, 1], dtype=float)

    # normalize to unit length
    vec1 = vec1 / np.linalg.norm(vec1)
    vec3 = vec3 / np.linalg.norm(vec3)
    vec4 = vec4 / np.linalg.norm(vec4)

    M1 = np.zeros((3,3),dtype=float) #rotation matrix

    # rotation matrix setup
    M1[:,0] = vec1
    M1[:,1] = vec3
    M1[:,2] = vec4

    print(M1)
    return

    # get the real part of the quaternion first
    r = math.sqrt(float(1)+M1[0,0]+M1[1,1]+M1[2,2])*0.5
    i = (M1[2,1]-M1[1,2])/(4*r)
    j = (M1[0,2]-M1[2,0])/(4*r)
    k = (M1[1,0]-M1[0,1])/(4*r)

    print(f"\"crossSectionOrientation\": [{i},{j},{k},{r}],")
    print("For zarr")
    print(f"\"crossSectionOrientation\": [0, 0, 0, -1],")
    print("Put dimensions at x,y,z,t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--degrees", help="Enter the angle in degrees", required=True, default=0.0, type=float)
    parser.add_argument("--axis", help="Enter the axis to rotate by (x, y, z)", required=True, default="x", type=str)
    args = parser.parse_args()
    degrees = args.degrees
    axis = args.axis

    #create_rotation_matrix(angle, rotateby)

    rigid_transform_3d(degrees, axis)
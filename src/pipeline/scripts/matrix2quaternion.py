import numpy as np
import math

theta = np.pi/4

rotateby = "z"

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

# get the real part of the quaternion first
r = math.sqrt(float(1)+M1[0,0]+M1[1,1]+M1[2,2])*0.5
i = (M1[2,1]-M1[1,2])/(4*r)
j = (M1[0,2]-M1[2,0])/(4*r)
k = (M1[1,0]-M1[0,1])/(4*r)

print("Quat: ",i,j,k,r)

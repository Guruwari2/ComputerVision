import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
#from matching_process import matching 

def find_epipoles(F):
    """
    Calculate epipoles from the fundamental matrix

    Args:
        F: 3x3 fundamental matrix

    Returns
        e0: epipole to image 1
        e1: epipole to image 2
    """
    #Find eigenvalues and eigenvectors of Fand F transposed
    value0, vector0 = np.linalg.eig(F)
    value1, vector1 = np.linalg.eig(np.transpose(F))

    #the epipoles are the eigenvector of the smallest eigenvalue
    e0 = vector0[:, np.argmin(value0)]
    e1 = vector1[:, np.argmin(value1)]

    return e0, e1


def rotation_matrix(u, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Args:
        axis:  3X1 numpy array
        theta: scalar. rotation angle
    Returns:
        R: rotation matrix
    """
    c = np.cos(theta);
    s = np.sin(theta);
    t = 1 - np.cos(theta);
    x = u[0];
    y = u[1];
    return np.array([[t*x*x + c, t*x*y, s*y],
                    [t*x*y, t*y*y + c, -s*x],
                    [-s*y, s*x, c]])

def find_prewarp(F):
    """
    Find H0 and H1 from the fundamental matrice F
    """
    #get first epipoles
    e0, e1 = find_epipoles(F)

    #axis of rotation
    d0 = np.array([-e0[1], e0[0], 0])

    #find corresponding axis in image 1
    Fd0 = F.dot(d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0])

    #find angle of rotation
    theta0 = np.arctan(e0[2]/(d0[1]*e0[0] - d0[0]*e0[1]))
    theta1 = np.arctan(e1[2]/(d1[1]*e1[0] - d1[0]*e1[1]))

    #rotation of angle theta about axis d
    R_d0_theta0 = rotation_matrix(d0, theta0)
    R_d1_theta1 = rotation_matrix(d1, theta1)


    #new epipoles
    new_e0 = R_d0_theta0.dot(e0)
    new_e1 = R_d1_theta1.dot(e1)

    #find new angle of rotation
    phi0 = -np.arctan(new_e0[1]/new_e0[0])
    phi1 = -np.arctan(new_e1[1]/new_e1[0])

    #rotation of angle phi about the zero point
    R_phi0 = np.array([[np.cos(phi0), -np.sin(phi0), 0],
                       [np.sin(phi0), np.cos(phi0), 0],
                       [0, 0, 1]])
    R_phi1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                       [np.sin(phi1), np.cos(phi1), 0],
                       [0, 0, 1]])

    H0 = R_phi0.dot(R_d0_theta0)
    H1 = R_phi1.dot(R_d1_theta1)

    return H0, H1



def normalize_points(points):
    mean = np.mean(points, axis=0)
    shifted_points = points - mean
    scale = np.sqrt(2) / np.mean(np.linalg.norm(shifted_points, axis=1))
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    normalized_points = (T @ np.column_stack((points, np.ones(points.shape[0]))).T).T
    return normalized_points[:, :2], T

def fundamental_matrix(points1, points2):
    points1_normalized, T1 = normalize_points(points1)
    points2_normalized, T2 = normalize_points(points2)

    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = points1_normalized[i]
        x2, y2 = points2_normalized[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0
    F_normalized = U @ np.diag(S) @ Vt

    F = T2.T @ F_normalized @ T1

    return F / F[2, 2]






def prewarp_proc(p1, p2):
    #points1, points2 = matching(image1,  image2)
    #F, mask = cv2.findFundamentalMat(p1,p2,cv2.RANSAC)
    F = fundamental_matrix(p1, p2)
    #print(F)
    H0, H1 = find_prewarp(F)


    #prewarp_img1 = cv2.warpPerspective(image1, H0, (image1.shape[0], image1.shape[1]))
    #prewarp_img2 = cv2.warpPerspective(image2, H1, (image2.shape[0], image2.shape[1]))
#
    #return prewarp_img1, prewarp_img2

    return H0, H1


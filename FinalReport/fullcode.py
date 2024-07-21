import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import imageio
import os


## Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_face_landmarks(image, detector = dlib.get_frontal_face_detector(), predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')):
#def get_face_landmarks(image):
    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Assuming the first detected face is the one we want to process
    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        return np.array(points)
    return None



# Plot images with landmarks using Matplotlib
def plot_landmarks(image, points, ax, title):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.scatter(points[:, 0], points[:, 1], c='red', s=10)
    ax.set_title(title)
    ax.axis('off')
    

# Morphing function
def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    im1_pts = np.array(im1_pts)
    im2_pts = np.array(im2_pts)
    intermediate_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts
    morphed_im = np.zeros(im1.shape, dtype=im1.dtype)

    for index,t in enumerate(tri.simplices):
        x1, y1, z1 = im1_pts[t].astype(np.float32)
        x2, y2, z2 = im2_pts[t].astype(np.float32)
        x, y, z = intermediate_pts[t].astype(np.float32)

        M1 = cv2.getAffineTransform(np.array([x1, y1, z1]), np.array([x, y, z]))
        M2 = cv2.getAffineTransform(np.array([x2, y2, z2]), np.array([x, y, z]))

        warped_im1 = cv2.warpAffine(im1, M1, (im1.shape[1], im1.shape[0]), flags=cv2.INTER_LINEAR)
        warped_im2 = cv2.warpAffine(im2, M2, (im2.shape[1], im2.shape[0]), flags=cv2.INTER_LINEAR)

        mask = np.zeros(im1.shape, dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(np.array([x, y, z])), (1.0, 1.0, 1.0), 16, 0)
        morphed_im = morphed_im * (1 - mask) + (1 - dissolve_frac) * warped_im1 * mask + dissolve_frac * warped_im2 * mask

    return morphed_im





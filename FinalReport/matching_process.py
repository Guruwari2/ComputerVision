import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import os


## Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def matching_face(image, detector = dlib.get_frontal_face_detector(), predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat'), nb_points = 68):
#def get_face_landmarks(image):
    # Detect faces in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Assuming the first detected face is the one we want to process
    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, nb_points):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        return np.array(points)
    return None

def matching_object(image1, image2):
    
    print("Matching Objects...")
    # Initialize SIFT detector
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Perform the matching
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    good_matches.sort(key=lambda x: x.distance)


        # Draw matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, good_matches[:19], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    # Display the image with matches using matplotlib
    #plt.figure(figsize=(20, 10))
    #plt.imshow(img_matches_rgb)
    #plt.axis('off')  # Hide the axis
    #plt.show()
#
    #Extract location of good matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches[:19]])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches[:19]])


    return points1, points2
    






window_name = "Select Points"
# Mouse callback function to capture points
def select_points(event, x, y, flags, param):
    global clicked_p1, clicked_p2, img_display, shape1
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < shape1:  # Check if the click is on the left image
            clicked_p1.append((x, y))
            cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
        else:  # The click is on the right image
            clicked_p2.append((x - shape1, y))
            cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)

        #img_display = cv2.resize(img_display, (960, 540)) 
        cv2.imshow(window_name, img_display)


# Function to add manual points
def add_manual_points(img1, img2):
    global img_display, clicked_p1, clicked_p2, shape1
    clicked_p1, clicked_p2 = [], []
    img_display = np.hstack((img1, img2))
    shape1= img1.shape[1]
    #img_display = cv2.resize(img_display, (1200, 700)) 
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, select_points)
    print("Click on the points in the images, press 'q' to finish.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    print("Manual Matching Points Selection Finished")
    return np.array(clicked_p1), np.array(clicked_p2)
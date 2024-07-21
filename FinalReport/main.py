import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import imageio
import re

from prewarp_process import prewarp_proc
from morphing_process import morphing
from postwarp_process import postwarp_proc
from matching_process import matching_face, matching_object, add_manual_points

class ViewMorphing(object):
    def __init__(self):
        pass


    def open_image(self, img1_path, img2_path = None, img1_name = 'Image1', img2_name = 'Image2'):

        if img1_path == None:
            print("Error: Could not load image.")
            return False

        elif img2_path == None :

            self.img1 = cv2.imread(img1_path)
            self.img2 = cv2.flip(self.img1, 1)
            self.common_size = self.img1.shape
            self.img1 = cv2.resize(self.img1, [500,500])
            self.img2 = cv2.resize(self.img2, [500,500])
            self.common_size = [500, 500]
            print("reading and flipping")
        else : 
            self.img1, self.img2 = cv2.imread(img1_path), cv2.imread(img2_path)
        
            self.height1, self.width1,self.c = self.img1.shape
            self.height2, self.width2, self.c = self.img2.shape
            
            self.common_size = (min(self.height1,self.height2), min(self.width1,self.width2))
            
            # Resize both images to the common size
            self.img1 = cv2.resize(self.img1, self.common_size)
            self.img2 = cv2.resize(self.img2, self.common_size)
    
            self.img1 = cv2.resize(self.img1, [500,500])
            self.img2 = cv2.resize(self.img2, [500,500])
            self.common_size = [500, 500]

        self.img1_name, self.img2_name = img1_name, img2_name
    
    def prewarp(self):
        #p1, p2 = matching_face(self.img1, nb_points = 8), matching_face(self.img2, nb_points=8)
        p1, p2 = matching_object(self.img1, self.img2)
        #p1, p2 = add_manual_points(self.img1,self.img2)

        H0, H1 = prewarp_proc(p1, p2)
        self.corners = np.array([[0,0],[self.common_size[0],0],[0,self.common_size[1]],[self.common_size[0],self.common_size[1]]], dtype=np.float32).reshape(-1, 1, 2)
        self.transformed_corners = cv2.perspectiveTransform(self.corners, H1).reshape(-1,2)

        max_x, max_y = np.max(self.transformed_corners, axis=0)        
        min_x, min_y = np.min(self.transformed_corners, axis=0)
        self.common_size = np.array([max_x-min_x, max_y-min_y], dtype=np.int32)

        self.img1 = cv2.warpPerspective(self.img1, H0, (self.common_size[0], self.common_size[1]))
        self.img2 = cv2.flip(self.img1, 1)
        #self.img1 = cv2.resize(self.img1, [500,500])
##
        #self.img2 = cv2.resize(self.img2, [500,500])
        #common_size = [500,500]

        #self.img2 = cv2.warpPerspective(self.img2, H1, (self.common_size[0]*4, self.common_size[1]*4))

        #self.img1, self.img2 = prewarp(p1, p2)

        cv2.imwrite('Prewarp/'+self.img1_name+'_prewarped.jpg', self.img1)
        cv2.imwrite('Prewarp/'+self.img2_name+'_prewarped.jpg', self.img2)



    def  find_matching_points(self, mp = False, face = True, add_corner = True):
        if face :
            self.points1, self.points2 = matching_face(self.img1), matching_face(self.img2)
        else :
            self.points1, self.points2 = matching_object(self.img1, self.img2)

        if mp ==True : # manual matching points
            self.mp1, self.mp2 = add_manual_points(self.img1, self.img2)
            self.points1, self.points2, self.mp1, self.mp2 = self.points1.tolist(), self.points2.tolist(), self.mp1.tolist(), self.mp2.tolist() 
            for mp1 in self.mp1:
                self.points1.append(mp1)
            for mp2 in self.mp2:
                self.points2.append(mp2)

            self.points1, self.points2 = np.array(self.points1), np.array(self.points2)

        if add_corner:
            self.corners = [[0,0],[self.common_size[0],0],[0,self.common_size[1]],[self.common_size[0],self.common_size[1]]]
            self.points1 = self.points1.tolist()
            self.points2 = self.points2.tolist()
            
            for corner in self.corners:
                self.points1.append(corner)
                self.points2.append(corner)
                
            self.points1, self.points2 = np.array(self.points1) , np.array(self.points2)
        self.tri1 = Delaunay(self.points1)

        #self.tri2 = Delaunay(self.points2)



    def morphing(self, warp_frac = 0.5, mp = False):
        self.warp_frac = warp_frac
        #self.find_matching_points(mp)
        self.img_morphed = morphing(self.img1, self.img2, self.points1, self.points2, self.tri1, warp_frac)
        cv2.imwrite('Morphed/Pictures_morphed_'+str(int(warp_frac*100))+'.jpg', self.img_morphed)

    def postwarp(self):
        self.img_final= postwarp_proc(self.img_morphed)
        cv2.imwrite('Results/final'+str(int(self.warp_frac*100))+'.jpg', self.img_final)



if __name__ == '__main__':
    import os

#################################################################################################################
    
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description='Process view morphing, gif creation and warping fraction ')
    
    # Add an optional boolean argument for gif creation
    parser.add_argument('--gif', action='store_true', help='A boolean flag for gif creation')
    
    # Add an optional boolean argument for view morphing
    parser.add_argument('--view', action='store_true', help='An boolean flag for view morphing')

    # Add an optional float argument for warping fraction
    parser.add_argument('--wfrac', type=float, default=0.5, help='A fraction value for morphing')

    # Add an optional boolean argument for manual points 
    parser.add_argument('--manualpoints', action='store_true', help='A boolean flag for adding manual matching points')

    #Add an optional boolean if it is an object or a face
    parser.add_argument('--noface', action='store_false', help='A boolean flag for adding manual matching points')

    #Add an optional boolean if it is an object or a face
    parser.add_argument('--nocorner', action='store_false', help='A boolean flag for adding manual matching points')

    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the values of x and y
    gif_creation = args.gif
    view_morph = args.view
    mp = args.manualpoints
    warp_frac = args.wfrac
    face = args.noface
    add_corner = args.nocorner

#################################################################################################################

    # Directory containing the images
    directory = 'input'

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    # List all files in the directory
    files = os.listdir(directory)
    files.sort(key=natural_sort_key)
    # Assuming the directory contains exactly two images
    if len(files) == 1:
        print("solo image")
        image1_path = os.path.join(directory, files[0])
        image2_path = None
    elif len(files) == 2:
        image1_path = os.path.join(directory, files[0])
        image2_path = os.path.join(directory, files[1])
    else:
        image1_path = 'test_input/professor.jpg'
        image2_path = 'test_input/student.jpg'

#################################################################################################################
    
    vm = ViewMorphing()



    print("Images Loading")
    vm.open_image(image1_path, image2_path)
    if view_morph:
        print("PreWarping in process")
        vm.prewarp()

    print("Finding Matching Points")

    vm.find_matching_points(mp, face = face, add_corner=add_corner,)


    if not gif_creation:
        print("Morphing in process with warp_value : " + str(warp_frac) )

        vm.morphing(warp_frac)

    if view_morph:
        print("postwarping in process")
        vm.postwarp()

    if gif_creation:
        print("Gif Creation in process \n" )

        gif_name = f"{vm.img1_name}_to_{vm.img2_name}.gif"
        gif_path = os.path.join('gif', gif_name)
        frames_num = 11
        second_each_frame = 1

        for frac in np.linspace(0, 1, frames_num):
            vm.morphing(frac)

        files = os.listdir("Morphed")
        files.sort(key=natural_sort_key)
        im_gif = []

        for image_name in files :
            image_path = os.path.join("Morphed", image_name)

            # Save to animated GIF
            im_gif.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        
            #images.append((morphed_im * 255).astype(np.uint8))
        
        imageio.mimsave(gif_path, im_gif, duration=second_each_frame)

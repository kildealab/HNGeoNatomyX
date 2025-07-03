"""
Created on January 2024 
@author: Odette Rios-Ibacache 

"""

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
import cv2, os, json
import numpy as np
from skimage import data
from skimage.morphology import disk
from skimage.filters import median
import pydicom as dcm
import rtdsm
from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat  
from skimage.morphology import black_tophat, skeletonize, convex_hull_image 
from skimage.morphology import disk 

def get_slice(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+'/'+f)
        positions.append(d.ImagePositionPatient) 
    positions = sorted(positions, key=lambda x: x[-1])
    return positions
    
def get_path_RS(path_patient):
    file_RS = sorted([x for x in os.listdir(path_patient) if 'RS' in x])[-1]
    full_path = os.path.join(path_patient, file_RS)
    return full_path

def get_start_position_dcm(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+'/'+f)
        positions.append(d.ImagePositionPatient)
    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = d.PixelSpacing
    
    return start_x, start_y, start_z, pixel_spacing 

def get_info_CT(path_patient):
    file_RS = sorted([x for x in os.listdir(path_patient) if x[9:11]=='CT' in x])[0]
    full_path = os.path.join(path_patient, file_RS)
    return full_path
    
def get_contour_mask(masked_img,h_center,k_center,min_radius,max_radius,thres_grey):
    positions = []

    values_angles = np.linspace(min_angle,max_angle,200)
    values_radius = np.linspace(min_radius,max_radius,200)
 
    for angle in values:
        count = 0 
        posx = 0
        posy = 0
        for r in sorted(values_radius)[::-1]:
            x = r*np.cos(angle*np.pi/180) + h_center
            y = r*np.sin(angle*np.pi/180)+ k_center
  
            if masked_img[int(y)][int(x)]==1:
                count = count +1
                if count==1:
                    posx = int(x)
                    posy = int(y)
                    positions.append([[posx,posy]])
                    
    mask_pointsx = []
    for point in positions:
        mask_pointsx.append(point[0][0])
    mask_pointsy = []
    for point in positions:
        mask_pointsy.append(point[0][1])    
    
    return positions,mask_pointsx,mask_pointsy
    
def add_coords(positions, ct_path,slice_number):
    
    mask_m = positions.copy()
    start_x, start_y, start_z, spacing,p = get_start_position_dcm(ct_path)
    x,y = get_ROI_pixel_array_m(mask_m,start_x,start_y,spacing)

    z = float(get_slice(ct_path)[slice_number][2])
    for x1 in range(0,len(x)):
        contour_mask.append([x[x1],y[x1],z])
    
def get_mask_nifti(roi_array,start_x,start_y,pixel_spacing):
    x = []
    y = []
  
    for i in range(0,len(roi_array)):
        x.append(((roi_array[i][0]/pixel_spacing[0]) - (start_x/pixel_spacing[0])))
        y.append(((roi_array[i][1]/pixel_spacing[1]) - (start_y/pixel_spacing[1])))
    return x, y
    
def get_ROI_pixel_array_m(roi_array,start_x,start_y,pixel_spacing):
    x = []
    y = []
    
    for i in range(0,len(roi_array)):

        x.append(((roi_array[i][0][0]) + start_x/pixel_spacing[0])*pixel_spacing[0])
        y.append(((roi_array[i][0][1]) + start_y/pixel_spacing[1])*pixel_spacing[1])
    return x, y


def save_mask_contour(contour_mask,patient_id_str):
    mesh = pv.PolyData(contour_mask)
    mesh.connectivity(largest=True)
    mask_json = {'Mask' : contour_mask}
    with open("registered/mask/Mask_"+str(patient_id_str)+".json", "w") as outfile:
        json.dump(mask_json, outfile)
    print('save at '+ "registered/mask/Mask_"+str(patient_id_str)+".json")
    return


def contour_slice(ct_files,ct_path,slice_value,h_center,k_center,min_radius,max_radius,thres_grey,plot=True):
    print('------- CURRENTLY WORKING ON SLICE NUMBER'+ str(j)+'------')
    
    points_to_use = []
    ct_file = ct_files[slice_value]
    img = ct_file.pixel_array

    if plot=True:
        plt.imshow(img,camp='gray')
        plt.title('Slice to be contour including BODY and TREATMENT MASK EDGE')
        plt.show()
        
    normalized_img = cv2.normalize(img, None, alpha = -200, beta = img.max(), norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    start_x, start_y, start_z, spacing = get_start_position_dcm(ct_path)
    body_contour= rtdsm.get_pointcloud('BODY', path_RS, False)[0]
    z_values_body_contour = body_contour[:,2]
    zs_sorted = list(dict.fromkeys(z_values_body_contour))
        
    body_slice = []
    for point in body_contour:
        if point[2]==zs_sorted[slice_value]:
            body_slice.append(point[0:2])
    
    points_x,points_y = get_mask_nifti(body_slice,start_x,start_y,spacing)
    slicee = list(zip(points_x,points_y))
    points_to_use.append(slicee)
    ctr = np.array(slicee).reshape((-1,1,2)).astype(np.int32)

    mask_zeros = np.zeros((512,512), np.uint8)
    cv2.fillPoly(mask_zeros,pts=ctr,color=(255,255,255))
    dilated = dilation(mask_zeros, disk(1))
    ret, thresh = cv2.threshold(dilated,0, 1, cv2.THRESH_BINARY)            
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    mask_zeros_2 = np.zeros((512,512), np.uint8)
    cv2.drawContours(mask_zeros_2, cnt, 0, 255, -1)

    img2 = normalized_img.copy()
    img2[normalized_img<thres_grey] = 0
    img2[normalized_img>=thres_grey] = 1
    
    img_dilated_2 = dilation(img2, disk(2))
    mask_c = np.zeros(img.shape[:2], dtype="uint8")
    
    #THE VALUES 11, AND 20 CAN BE CHANGED DEPENDING ON THE RESULTS. THESE VALUES ARE USED TO REMOVE THE TREATMENT TABLE
    mask_c[0:int(np.max(np.array(points_to_use)[0][:,1]))-11, :] = 255
    
    mask_r = np.zeros(img.shape[:2], dtype="uint8")
    mask_r[0:int(np.max(np.array(points_to_use)[0][:,1]))-11, int(np.min(np.array(points_to_use)[0][:,0]))-20:int(np.max(np.array(points_to_use)[0][:,0]))+20] = 255
    
    masked_img = cv2.bitwise_and(img_dilated_2,img_dilated_2,mask = mask_c)
    
    disk_radius_1 = disk(1)
    img_erosed = erosion(masked_img, disk_radius_1)
    img_dilated_6 = dilation(img_erosed, disk(6))
    disk_radius_2 = disk(2)
    
    img_erosed_2 = erosion(img_dilated_6, disk_radius_2)
    masked_img2 = cv2.bitwise_and(img_erosed_2,img_erosed_2,mask = mask_r)

    if plot=True:
        plt.imshow(masked_img2,camp='gray')
        plt.title('Mask to be contoured including BODY and TREATMENT MASK EDGE')
        plt.show()
    
    positions,mask_pointsx,mask_pointsy = get_contour_mask(masked_img2,h_center,k_center,min_radius,max_radius,thres_grey)

    if plot==True:
        plt.imshow(masked_img2,camp='gray')
        plt.title('Mask to be contoured including BODY and TREATMENT MASK EDGE')
        plt.scatter(mask_pointsx,mask_pointsy,c='r',s=1)
        plt.show()

    return positions

######################


#path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/' # Path to patient directories
patients = os.listdir(path_CBCTs)
files = [f for f in os.listdir(path_CBCTs) if os.path.isfile(f)]
patients_path = [path_CBCTs+patient+"/" for patient in patients]

patient_id_str = insert('Insert patient number: ')

ct_path = get_info_CT(patients_path[patients.index(patient_id_str)])
ct_files = [pydicom.dcmread(os.path.join(ct_path, f)) for f in os.listdir(ct_path) if 'CT' in f]
path_RS = get_path_RS(ct_path)

# Sort the CT files in ascending order by their image position (z-axis)
ct_files.sort(key=lambda x: x.ImagePositionPatient[2])
ct_file = ct_files[0]
contour_mask = []

#e.g. h_center,k_center,min_angle,max_angle,min_radius,max_radius,thres_grey = 250,350,-175,-5,55,250,60


#MINIMUM SLICE WHERE YOU CAN VISUALIZE THE MASK. THIS VALUE CAN BE CHANGED DEPENDING ON EACH IMAGE.
minimum_slice_with_mask= 20
for slice_value in range(minimum_slice_with_mask,len(ct_files)):

    contour_slice(ct_files,ct_path,slice_value,h_center,k_center,min_radius,max_radius,thres_grey)
    add_coords(positions, ct_path,slice_value)
    
save_mask_contour(contour_mask,patient_id_str)

    

    


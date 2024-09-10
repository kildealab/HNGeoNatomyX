from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
import cv2
import pydicom
import os
import numpy as np
from skimage import data
from skimage.morphology import disk
from skimage.filters import median
import pydicom as dcm
import numpy as np
import rtdsm
import cv2
import json
from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk 


def get_slice(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+'/'+f)
        positions.append(d.ImagePositionPatient) 
    
    positions = sorted(positions, key=lambda x: x[-1])
 
    return positions
    
def get_path_RS(path_k):
    path_patient = path_k
   
    file_RS = sorted([x for x in os.listdir(path_patient) if 'RS' in x])[-1]
    print(file_RS)
    path2 = os.path.join(path_patient, file_RS)
 
    return path2

def get_start_position_dcm(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = dcm.dcmread(CT_path+'/'+f)
        
        positions.append(d.ImagePositionPatient)
        #print(CT_path+'/'+f) 
    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = d.PixelSpacing
    
    return start_x, start_y, start_z, pixel_spacing 

def get_info2(path_k):
    path_patient = path_k
    file_RS = sorted([x for x in os.listdir(path_patient) if x[9:11]=='CT' in x])[0]
    print(sorted([x for x in os.listdir(path_patient) if x[9:11]=='CT' in x]))

    path2 = os.path.join(path_patient, file_RS)
    files = [x for x in os.listdir(path2) if 'CT' in x]
    files2 = []
    
    return path2
    
def contour_mask(slice_im,ct_files,h,k,a,b,r1,r2,thres):
    j = slice_im
    ct_file = ct_files[j]
    img = ct_file.pixel_array

    img3 =img.copy()
    img3[img<thres]=0
    img3[img>=thres]=1
    maslcirc = []
    pos = []
    values = np.linspace(a,b,200)
    values2 = np.linspace(r1,r2,200)
    
    for t in values:

        count = 0 
        posx = 0
        posy = 0
        for r in sorted(values2)[::-1]:
            x = r*np.cos(t*np.pi/180) + h 
            y = r*np.sin(t*np.pi/180)+ k 
  
            if img3[int(y)][int(x)]==1:
                count = count +1
                if count==1:
                    posx = int(x)
                    posy = int(y)
   
                    pos.append([[posx,posy]])
    mask2x_m = []
    for p in pos:
        mask2x_m.append(p[0][0])
    mask2y_m = []
    for p in pos:
        mask2y_m.append(p[0][1])
    
   # fx,ax = plt.subplots(1,2,figsize=(10,4))
   # ax[0].imshow(img3)
#    ax[0].scatter(mask2x_m,mask2y_m,c='r',s=1)
 #   ax[1].imshow(img)
 #   ax[1].scatter(mask2x_m,mask2y_m,c='r',s=1)
  #  plt.show()
    
    return pos
    
def add_coords(pos, ct_path,j):
    
    mask_m = pos.copy()
    start_x, start_y, start_z, spacing,p = get_start_position_dcm(ct_path)
    x,y = get_ROI_pixel_array_m(mask_m,start_x,start_y,spacing)

    z = float(get_slice(ct_path)[j][2])
    for x1 in range(0,len(x)):
            #    z = ((j + start_z/3)*3) 
        contour_m.append([x[x1],y[x1],z])


def get_mask_nifti(roi_array,start_x,start_y,pixel_spacing):
    x = []
    y = []
  
    for i in range(0,len(roi_array)):
        #print(roi_array[i])
        
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
    
######################

PATH = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/' # Path to patient directories
patients = os.listdir(PATH)
files = [f for f in os.listdir(PATH) if os.path.isfile(f)]
patients_path = [PATH+patient+"/" for patient in patients]

t = insert('Insert patient number: ')

ct_path2 = get_info2(patients_path[patients.index(t)])

ct_files = [pydicom.dcmread(os.path.join(ct_path2, f)) for f in os.listdir(ct_path2) if 'CT' in f]

# Sort the CT files in ascending order by their image position (z-axis)
ct_files.sort(key=lambda x: x.ImagePositionPatient[2])
ct_file = ct_files[0]



#j = slice_im
import numpy as np
import rtdsm
import cv2
from matplotlib import pyplot as plt

from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk 


h,k,a,b,r1,r2,thres1 = 250,350,-175,-5,55,250,60
for j in range(20,len(ct_files)):
    print(j)
    points_to_use = []
    ct_file = ct_files[j]
    img = ct_file.pixel_array
    n_img = cv2.normalize(img, None, alpha = -200, beta = img.max(), norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    start_x, start_y, start_z, spacing,positions = get_start_position_dcm(ct_path2)
    mR= rtdsm.get_pointcloud('BODY', path_RS, False)[0]
    z = mR[:,2]
    zs = list(dict.fromkeys(z))
    #for j in range(0,len(zs)):
    body_slice = []
    for p in mR:
        if p[2]==zs[j]:
            body_slice.append(p[0:2])
    
    px,py = get_mask_nifti(body_slice,start_x,start_y,spacing)
    slicee = list(zip(px,py))
    
    points_to_use.append(slicee)

    ctr = np.array(slicee).reshape((-1,1,2)).astype(np.int32)
            
    mask2 = np.zeros((512,512), np.uint8)
    cv2.fillPoly(mask2,pts=ctr,color=(255,255,255))
    dilated = dilation(mask2, disk(1))
    ret, thresh = cv2.threshold(dilated,0, 1, cv2.THRESH_BINARY)
            
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    mask22 = np.zeros((512,512), np.uint8)
    cv2.drawContours(mask22, cnt, 0, 255, -1)
  
            #cv2.drawContours(mask, cnt, 0, 1, -1)
    
    
    #plt.imshow(mask22)
    #plt.show()
        
    img2 =n_img.copy()
    img2[n_img<thres1]=0
    img2[n_img>=thres1]=1
    
    
    #norm_image = cv2.normalize(img3, None, alpha = -1000, beta = img.max(), norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #img_s = show_slice_window(norm_image,50, img.max())
    
    #clp = intensity_seg(img_s, 200, 290)
    #med = median(img_s, disk(5))
    #print(int(np.max(np.array(points_to_use)[0][:,1])))
    img22 = dilation(img2, disk(2))
    mask_c = np.zeros(img.shape[:2], dtype="uint8")
    mask_c[0:int(np.max(np.array(points_to_use)[0][:,1]))-11, :] = 255
    
    mask_r = np.zeros(img.shape[:2], dtype="uint8")
    mask_r[0:int(np.max(np.array(points_to_use)[0][:,1]))-11, int(np.min(np.array(points_to_use)[0][:,0]))-20:int(np.max(np.array(points_to_use)[0][:,0]))+20] = 255
    
    
    masked_img = cv2.bitwise_and(img22,img22,mask = mask_c)
    
    footprint = disk(1)
    img3 = erosion(masked_img, footprint)
    #med = median(img3, disk(1))
    img4 = dilation(img3, disk(6))
    footprint = disk(2)
    
    img5 = erosion(img4, footprint)
    masked_img2 = cv2.bitwise_and(img5,img5,mask = mask_r)
    
       
    maslcirc = []

    pos = []
    values = np.linspace(a,b,200)
    values2 = np.linspace(r1,r2,200)
 
    for t in values:
        count = 0 
        posx = 0
        posy = 0
        for r in sorted(values2)[::-1]:
            x = r*np.cos(t*np.pi/180) + h 
            y = r*np.sin(t*np.pi/180)+ k 
  
            if masked_img2[int(y)][int(x)]==1:
                count = count +1
                if count==1:
                    posx = int(x)
                    posy = int(y)
   
                    pos.append([[posx,posy]])
    mask2x_m = []
    for p in pos:
        mask2x_m.append(p[0][0])
    mask2y_m = []
    for p in pos:
        mask2y_m.append(p[0][1])
    
    mask_t = np.zeros(img.shape[:2], dtype="uint8") 
    cv2.drawContours(mask_t, np.array(pos), 0, 255, -1)
    
    #plt.imshow(masked_img)
    #plt.show()
    plt.imshow(masked_img2)
    #plt.plot(np.array(points_to_use[0][:,0]),np.array(points_to_use)[0][:,1],'ro')
    plt.scatter(mask2x_m,mask2y_m,c='r',s=1)
    plt.show()
    add_coords(pos, ct_path2,j)
    
    #fx,ax = plt.subplots(1,2,figsize=(10,4))
    #ax[0].imshow(img3)
    #ax[0].scatter(mask2x_m,mask2y_m,c='r',s=1)
    #ax[1].imshow(img)
    #ax[1].scatter(mask2x_m,mask2y_m,c='r',s=1)
    #plt.show()
    
#import pyvista as pv

mesh = pv.PolyData(contour_m)
mesh.connectivity(largest=True)
pati = '671'
mask3 = {'Mask' : contour_m}
with open("registered/mask/Mask_"+pati+".json", "w") as outfile:
    json.dump(mask3, outfile)
print('save at '+ "registered/mask/Mask_"+pati+".json")
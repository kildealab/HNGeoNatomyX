"""
Created on Jul 2024 
@author: Odette Rios-Ibacache and James Manalad

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import scipy
import json
from scipy.stats import sem
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
import numpy as np
import gc, os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d
from datetime import date
import cv2
from pyvista import Cylinder
import alphashape
import pandas as pd
import pyvista as pv
import pydicom
import sympy as sym
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from skimage.draw import polygon
import random
from shapely import Polygon, intersection
from scipy.spatial.distance import directed_hausdorff
import point_cloud_utils as pcu



'''Gets the RS path from the RT structure file.  pat_id  is the patient ID and path_src is the path of the folder'''
def get_path_RS(pat_id, path_src):   
    path_patient = os.path.join(path_src, pat_id)  #FORMAT: path_src + '/'+pat_id  
    file_RS = [x for x in os.listdir(path_patient) if 'RS' in x][0]  # finds the RS file with the name RS.######.dcm
    return os.path.join(path_patient, file_RS)

'''Gets the RS path (Rt structure file) for the CT folder'''
def get_path_RS_CT(path_CT):  
    file_RS = [x for x in os.listdir(path_CT) if 'RS' in x][0]
    return os.path.join(path_CT, file_RS)
    
'''Gets the body contour key with the labels 'body' in it'''
def get_body_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'body' in x.lower()]

'''Gets the ROI key labels available in the RT dicom file'''
def get_ROI_keys(RS_file_path):  #Obtiene el ROI image
    RS_file = pydicom.read_file(RS_file_path)
    contour_keys = RS_file.StructureSetROISequence
    return [str(x.ROIName) for x in contour_keys]

'''Gets the PTV label in the keys available in the RT dicom file'''
'''NOTE: the PTV commonly use is the PTV_All in the RT file'''
def get_PTV_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'ptv' in x.lower()]

'''Sorts the key from least to greatest body keys'''
def sort_body_keys(keys_body): 
    new_keys_body = []
    nums = []
    for key in set(keys_body):
        str_frac_num = key.split('-')[-1]
        if not str_frac_num.lower() == 'body':
            nums.append(int(str_frac_num))
        else:
            new_keys_body.append(key)
    nums = sorted(nums)
    for num in nums:
        for key in keys_body:
            if str(num) == key.split('-')[-1]:
                new_keys_body.append(key)    
    return new_keys_body
    
#-------------------------------------
'''Gets the resolution in the z direction (height), i.e. the CT slice thickness'''
def get_contour_z_spacing(contours):
    z_vals = np.array(list(set(contours[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    diff_arr = sorted_z[:-1] - sorted_z[1:]
    #print(abs(np.mean(diff_arr)))
    return abs(np.mean(diff_arr))

'''Gets the maximum and minimum z values from given contours
and then cuts them within a selected range and a margin of 
3 times the z spacing'''
def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    return contours_1, contours_2
   
def trim_contours_to_match_zs_edge(contours_1,contours_2,z_min,z_max): 
    max_z = z_max 
    min_z = z_min 
        
    contours_1 = np.array([x for x in contours_1 if x[2] <= max_z and x[2] >= min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] <= max_z and x[2] >= min_z])
    
    return contours_1,contours_2   
    
'''Gets the surface by using the marching cubes methods and the rtdsm function
see skidimage marching cubes webpage for more details'''
def get_surface_marching_cubes(contours,IMG_RES):
    img_res = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours)]
    verts, faces, pvfaces = rtdsm.get_cubemarch_surface(contours.copy(), img_res)
    mesh = pv.PolyData(verts, faces=pvfaces)
    return mesh.extract_surface()

def get_surface_marching_cubes_clusters(contours):
    img_res = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours)]
    verts, faces, pvfaces = rtdsm.get_cubemarch_surface_clusters(contours.copy(), img_res)
    mesh = pv.PolyData(verts, faces=pvfaces)
    return mesh.extract_surface()

'''Split a point cloud given a surface'''
def split_cloud_by_surface(cloud, surface):
    cloud.compute_implicit_distance(surface, inplace=True)
    inner_cloud = cloud.threshold(0.0, scalars="implicit_distance", invert=True)  
    outer_cloud = cloud.threshold(0.0, scalars="implicit_distance", invert=False)
    return inner_cloud, outer_cloud

'''For a given set of contours and x value, it returns
the closest x values that are in one of the contour points'''
def get_closest_x(x, contours):
    closest_x = contours[0][0]
    min_abs_diff = 10000
    for point in contours:
        current_x = point[0]
        abs_diff = abs(current_x - x)
        if abs_diff < min_abs_diff:
            min_abs_diff = abs_diff
            closest_x = current_x
    return closest_x


# For a given set of contours and x value, return point with maximum y value
def get_point_with_max_y_around_given_x(x, contours):
    target_x = x
    max_y = -1
    for point in contours:
        current_x, current_y = point[0:2]
        if abs(current_x - x) < 0.5:
            if current_y > max_y:
                max_y = current_y
                target_x = current_x
    return (target_x, max_y)

'''Gets maximum and minimum values in a given contour (bounding box)'''
def get_bounding_box_dimensions(contours):
    max_x = max(contours[:,0])
    min_x = min(contours[:,0])
    diff_x = max_x - min_x
    max_y = max(contours[:,1])
    min_y = min(contours[:,1])
    diff_y = max_y - min_y
    
    return [diff_x, diff_y]
    

'''Gets the Reconstruction Diameter from the CBCT dicom files'''
#BE AWARE THAT THE RECONSTRUCTION DIAMETER SHOULD BE THE SAME FOR ALL THE CBCT IMAGES
def get_info_fov(path_patient):
    file_RS = [x for x in os.listdir(path_patient) if 'kV' in x][0]
    path2 = os.path.join(path_patient, file_RS)
    files = [x for x in os.listdir(path2) if 'CT' in x]
    files2 = []
    for j in files:
        files2.append(os.path.join(path2, j))
    dc_file = pydicom.read_file(files2[0])
    return dc_file.ReconstructionDiameter

def get_info_fov_minimum(patient,body_keys,pathpatient):
    file_RS = sorted([x for x in os.listdir(path_patient) if 'kV' in x])
    rs = []
    for p in range(0,len(body_keys)):
        path2 = os.path.join(path_patient, file_RS[p])
        files = [x for x in os.listdir(path2) if 'CT' in x]
        files2 = []
        for j in files:
            files2.append(os.path.join(path2, j))
        dc_file = pydicom.read_file(files2[0])
        r = dc_file.ReconstructionDiameter
        rs.append(float(r))
    return min(rs)
    
def centers(x1, y1, x2, y2, r):
    q = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2
    xx = (r ** 2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r ** 2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    return ((x3 + xx, y3 + yy))
                  
def get_estimate_center(body2,r):
    d_2 = pv.PolyData(body2).connectivity(largest=True)
    
    max_x = max(d_2.points[:,0])
    min_x = min(d_2.points[:,0])
    max_y = max(d_2.points[:,1])
    
    h = np.mean([max_x,min_x])
    y1 = get_point_with_max_y_around_given_x(max_x,d_2.points)
    y2 = get_point_with_max_y_around_given_x(min_x,d_2.points)
    k1 = y1[1] - np.sqrt(np.abs((r*0.5)**2 - (max_x-h)**2))
    k2 = y2[1] - np.sqrt(np.abs((r*0.5)**2 - (min_x-h)**2))
    
    y9 = get_point_with_max_y_around_given_x(max_x/4,d_2.points)
    y10 = get_point_with_max_y_around_given_x(min_x/4,d_2.points)
    y3 = get_point_with_max_y_around_given_x(max_x/2,d_2.points)
    y4 = get_point_with_max_y_around_given_x(min_x/2,d_2.points)
    y5 = get_point_with_max_y_around_given_x(0,d_2.points)
    y6 = get_point_with_max_y_around_given_x(max_x*7/8,d_2.points)
    y7 = get_point_with_max_y_around_given_x(min_x*7/8,d_2.points)
    
    k3 = y3[1] - np.sqrt(np.abs((r*0.5)**2 - (max_x/2-h)**2))
    k4 = y4[1] - np.sqrt((r*0.5)**2 - (min_x/2-h)**2)
   
    k6 = y6[1] - np.sqrt(np.abs((r*0.5)**2 - (max_x*7/8-h)**2))
    k7 = y7[1] - np.sqrt(np.abs((r*0.5)**2 - (min_x*7/8-h)**2))
    
    theta = np.linspace(0,2*np.pi,300)
 
    radius = r*0.5
    k = np.mean([k1,k2,k3,k4,k6,k7])
   
    xx1,yy1 = y6 
    xx2,yy2 = y7
    
    xx3,yy3 = y1
    xx4,yy4 = y2
    
    xx5,yy5 = y3
    xx6,yy6 = y4

    xx9,yy9 = y9
    xx10,yy10 = y10
    xx55,yy55 = y5    

    cp = centers(xx9,yy9,xx4,yy4,r*0.5)
    cd = centers(xx2,yy2,xx4,yy4,r*0.5)

    chh = centers(xx9,yy9,xx2,yy2,r*0.5)

    hc = np.mean([chh[0],cp[0],cd[0]])
    kc = np.mean([cp[1],chh[1],cd[1]])
    
    h2 = hc
    k2 = np.max([kc,k])
    return h2,k2
    
def get_elongation_only_central(body,z_m,z_m2,contour0):
    points_xy = []
    for j in body:
        if j[2]==z_m:
            points_xy.append([j[0],j[1]])
            
    points_xy_0 = []
    for j in contour0:
        if j[2]==z_m2:
            points_xy_0.append([j[0],j[1],j[2]])

    isox = np.mean([max(np.array(points_xy_0)[:,0]),min(np.array(points_xy_0)[:,0])])
    isoy = np.mean([max(np.array(points_xy_0)[:,1]),min(np.array(points_xy_0)[:,1])])

    iso = isox,isoy

    dists = np.sqrt((np.array(points_xy)[:,0]-isox)**2+(np.array(points_xy)[:,1]-isoy)**2)
    R_mean = (np.mean(dists))

    R_min = (np.min(dists))
    R_max = (np.max(dists))

    return R_max,R_min,R_max/R_min,R_mean

def search_cuts_z(contours):
    z_maxs = []
    z_mins = []
    for j in contours:
        z_maxs.append(max(j[:,2]))
        z_mins.append(min(j[:,2]))
    return max(z_mins),min(z_maxs)
    
#IMPORTANT: SOME BODIES PRESENT EDGES AT THE BOTTOM DUE TO THE AUTOMATIC CONTOURING, TO AVOID THOSE EDGES WE 
#TRIM THE BODIES +3 MM AT THE BOTTOM AND -3 MM AT THE TOP!
def trim_contours_to_match_zs(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2
    

def get_CT_CBCT_equal_body(body2,body1,z_max,z_min,h,k,r,IMG_RES):
    body_2 = pv.PolyData(body2).connectivity(largest=True)

    d1 = pv.PolyData(body1).connectivity(largest=True)

    bx = d1.points[:,0]
    by = d1.points[:,1]
    bz = d1.points[:,2]

    indexes = (bx-h)**2+(by-k)**2<=(r*0.8)**2

    bx2 = bx[indexes==True]
    by2 = by[indexes==True]
    bz2 = bz[indexes==True]

    points22 = list(zip(bx2,by2,bz2))
    d11 =pv.PolyData(points22)

    theta = np.linspace(0,2*np.pi,300)

    x = r*0.5 * np.cos( theta ) +h
    y = r*0.5 * np.sin( theta ) +k

    body_crop = [] #array points (cloud)
    zs = [i for i in d11.points[:,2]]
    zss = sorted(list(dict.fromkeys(zs)))
    for z in zss:
        ptosxy = []
        for p in d11.points:
            if z==p[2]:
                ptosxy.append([float(p[0]),float(p[1])])
        #pp1= Polygon(ptosxy)
        pp1= Polygon(ptosxy).buffer(0)
        pp2 = Polygon(list(zip(x,y)))
        pp3 = intersection(pp1,pp2)
        try:
            bx,by = pp3.exterior.coords.xy[0],pp3.exterior.coords.xy[1]
        except:
            coords = [[len(list(x.exterior.coords)),list(x.exterior.coords)] for x in pp3.geoms]
            bx,by = np.array((sorted(coords)[-1][-1]))[:,0],np.array((sorted(coords)[-1][-1]))[:,1]

        for j in range(0,len(bx)):
            body_crop.append((float(bx[j]),float(by[j]),float(z)))

    bbody = pv.PolyData(body_crop)
    bbody2,bbody1 = trim_contours_to_match_zs(body_2.points, bbody.points,z_min,z_max)
    s_body22 = get_surface_marching_cubes(bbody2,IMG_RES)
    s_body2 = pv.PolyData(s_body22).connectivity(largest=True)

    s_body11 = get_surface_marching_cubes(bbody1,IMG_RES)
    s_body1 = pv.PolyData(s_body11).connectivity(largest=True)
    return s_body1,s_body2

def get_keysall(patient,path_RS):
    pat_h = []
    #keys_body = get_body_keys(path_RS)
    #sorted_keys_body = sort_body_keys(keys_body)
    ROI_keys = get_ROI_keys(path_RS)
        
    return ROI_keys
    
def get_min_mandible_slice(body,mandible):
    minimum_mandible = min(mandible[:,2])
    zs_body = (body)[:,2]
    zs_body_clean = zs_body[~(np.isnan(zs_body))]
    z_position = np.argmin(abs(zs_body_clean - minimum_mandible))
    resulted_slice = zs_body_clean[z_position]
    return resulted_slice

def get_min_mandible_slice_from_surface(s_body1,mandible_contour):
    zs_body = s_body1[:,2].copy()
    zs_body_clean = zs_body[~(np.isnan(zs_body))]
    zs_mandible =  mandible_contour[:,2].copy()
    z_min_mandible = min(zs_mandible)
    z_position = np.argmin(abs(zs_body_clean-z_min_mandible)) 
    resulted_slice =  zs_body_clean[z_position]
    return resulted_slice

def translation_z(structure,z_value):
    matrix = np.array(([1,0,0,0],[0,1,0,0],[0,0,1,-z_value]))
    new = []
    for j in structure:
        new.append(list(j[0:3])+[1])
    new2 = []
    for j in new:
        new2.append(matrix@j)

    new_clean = []
    for j in new2:
        new_clean.append(list(j))
    return new_clean

def change_z_coordinates(structure,z_value):
    new_structure = translation_z(structure,z_value)
    return new_structure

def get_x_max(y, points):
    max_x = -10000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_y,1) == round(y,1):
            if current_x>max_x:
                max_x = current_x
    return max_x

'''For a given set of contours and x value, it returns the maximum y value'''
def get_max_y(x, contours):
    target_x = get_closest_x(x, contours)
    max_y = -1
    for point in contours:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(target_x,1):
            if current_y > max_y:
                max_y = current_y
    return max_y

    
def get_y_min(x, points):
    min_y = 10000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(x,1):
            if current_y<min_y:
                min_y = current_y
    return min_y
    
    
def get_x_min(y, points):
    min_x = 1000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_y,1) == round(y,1):
            if current_x<min_x:
                min_x = current_x
    return min_x

def get_length_lxy(body,z_min):

    spacing = get_contour_z_spacing(body.points)
    points_xy = []
    
    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])
            
    min_x = min(np.array(points_xy)[:,0])
    max_x = max(np.array(points_xy)[:,0])
    
    min_y = min(np.array(points_xy)[:,1])
    max_y = max(np.array(points_xy)[:,1])
    
    point1 = (min_x, get_max_y(min_x, points_xy))
    point3 = (get_x_min(max_y,points_xy),max_y)
    point5 = (max_x, get_max_y(max_x, points_xy))
    point6 = (get_x_min(min_y, points_xy),min_y)
    
    x1, y1 = point1
    x3, y3 = point3
    x5, y5 = point5
    x6,y6 = point6

    lx = np.sqrt((x1-x5)**2+(y1-y5)**2)
    ly = np.sqrt((x3-x6)**2+(y3-y6)**2)
            
    return lx,ly

def get_body_keys_not_RS(file_list):
    body_keys = []
    for k in file_list:
        key = k.split('.')[0]
        body_keys.append(key)

    sorted_keys = sort_body_keys(body_keys)
    return sorted_keys

#############################################
# NECK RELATED METRICS
############################################
def trim_contours_to_match_zs_neck(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV

    spacing_z = get_contour_z_spacing(contours_1)

    max_z = z_max 
    min_z = z_min + 3*spacing_z

    contours_1 = np.array([x for x in contours_1 if x[2] <= max_z and x[2] >= min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] <= max_z and x[2] >= min_z])
    return contours_1, contours_2
    
def get_z_bottom_neck(z_min,key_bodies_to_save,contours):
    if z_min>=-27:
        z_neck = -27
        return z_neck
    else:
        neck_zss = []
        for key_body_n in range(1,len(key_bodies_to_save)):
            body = contours[key_body_n]
            gc.collect()
              
            h,k = get_estimate_center(body,r)
            min_z = get_z_out_fov(body,h,k,r,z_min)
            neck_zss.append(min_z)
       
        if min(neck_zss)>=-27:
            z_neck = -27
            return z_neck
        else:
            z_neck = min(neck_zss)
            return z_neck

def get_surface_area(contours_body,IMG_RES):
    surface_body = get_surface_marching_cubes(contours_body,IMG_RES).smooth(n_iter=0)
    s_area = surface_body.area
    return s_area


def get_area_across_slices(key_body_n,str_pat_id,s_body,path_CBCTs):
    CT_path = get_info_replanned(str_pat_id,1,path_CBCTs)
                
    start_x, start_y, start_z, pixel_spacing = get_start_position_dcm(CT_path)
    s_body1 = get_surface_marching_cubes(s_body,IMG_RES[0:2])
    zs = sorted(list(set(s_body1.points[:,2])))
    areas = []
    for z in zs:
      ptosxy = get_contour_slice(s_body1,z)
                
      slice1 = np.array(ptosxy)[:,0:2]
    
      area = get_area(slice1,start_x,start_y,IMG_RES[0:2])
      areas.append(area)
    return np.mean(areas)

def get_elongation3D(body,body0):

    x_min = min(np.array(body0)[:,0])
    x_max = max(np.array(body0)[:,0])

    y_min = min(np.array(body0)[:,1])
    y_max = max(np.array(body0)[:,1])
    
    z_min = min(np.array(body0)[:,2])
    z_max = max(np.array(body0)[:,2])

    central_z = np.mean([z_min,z_max])
    central_x = np.mean([x_min,x_max])
    central_y = np.mean([y_min,y_max])
    
    dists = np.sqrt((np.array(body)[:,0]-central_x)**2+(np.array(body)[:,1]-central_y)**2+(np.array(body)[:,2]-central_z)**2)
    
    R_mean = (np.mean(dists))

    R_min = (np.min(dists))
    R_max = (np.max(dists))

    return R_max,R_min,R_max/R_min,R_mean
    
#CHECK FORMAT TYPE OF THE CONTOUR FILES SAVED, IF THEY ARE SAVED SEPARATELY
#IT CAN BE .dcm and .nii
def get_format(file_list):
    formatt = file_list[0].split('.')[1]    
    return format

#GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
def get_name_files(patient_path):
    replan = False
    CT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'CT' and len(d) == 23]
    CT_list.sort()

    CBCT_list_replan = []
    CBCT_list = [d for d in os.listdir(patient_path) if d[9:11] == 'kV']
    CBCT_list.sort()

    if len(CT_list) == 0:
        raise NotADirectoryError('No CT directories were found. Please ensure the following naming convention was used: "YYYYMMDD_CT_DD_MMM_YYYY".')

    elif len(CT_list) > 1: # Set replan to true if > 1 CT
        replan = True
    if replan==True:
        date_replan = CT_list[1][0:8]
        
        CBCT_list_replan = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) > int(date_replan)]
        CBCT_list_same_date = [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) == int(date_replan)]
        CBCT_list =  [CBCT for CBCT in CBCT_list if int(CBCT[0:8]) < int(date_replan)]

        # Organizing CBCTs with same date as replan CT into pre-post replan		
        for CBCT in CBCT_list_same_date:
            fx = CBCT.split('_')[-1][:-1]
            if int(fx) > int(CBCT_list[-1].split('_')[-1][:-1]):
                CBCT_list.append(CBCT)
            else:
                CBCT_list_replan.insert(0,CBCT)
                
        return CT_list[0],CBCT_list
    else:
        return CT_list[0],CBCT_list

'''Gets the key for the mandible RT structure (contour)'''
def get_key_mandible(path_RS0):
  
    keys = get_ROI_keys(path_RS0)

    key = [i for i in keys if 'MANDIBLE' in i or 'Mandible' in i or 'Bone_Mandible' in i]
    key_f = [y for y in key if '_opt' in y or 'z_' in y]
    key_ff = [i for i in key if i not in key_f]

    return key_ff[0]
    
   
def get_info_replanned(patient,index,path_CBCT_images):
    patient_path  = path_CBCT_images + patient+'/'
    CT, CBCT_list = get_name_files(patient_path)
    CTs_names = [CT]+CBCT_list
    path_complete = patient_path+ CTs_names[index]
    return path_complete

def get_mask_nifti(roi_array,start_x,start_y,pixel_spacing):
    '''
    Get the pixel positions (rather than the x,y coords) of the contour array so it can be plotted.
    '''
    x = []
    y = []
    
    for i in range(0,len(roi_array)):
        x.append((roi_array[i][0]/pixel_spacing[0]) - start_x/pixel_spacing[0])
        y.append((roi_array[i][1]/pixel_spacing[1]) - start_y/pixel_spacing[1])
        
    return x, y
    
def get_start_position_dcm(CT_path):
    positions = []
    for f in [file for file in os.listdir(CT_path) if 'CT' in file]:
        d = pydicom.dcmread(CT_path+'/'+f)
        
        positions.append(d.ImagePositionPatient)
 
    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = d.PixelSpacing
    
    return start_x, start_y, start_z, pixel_spacing

 
def get_center_fov(path_CBCT_images,str_pat_id):
    isos = rtdsm.get_pointcloud('AcqIsocenter', path_CBCT_images+'/'+str_pat_id+'/iso.dcm', False)[0]
    h = isos[0][0]
    k = isos[0][1]

    return h,k

################################
# TREATMENT MASK METRICS RELATED FUNCTIONS
#####################

def get_treatment_mask_contour(path_treatment_masks,str_pat_id):
    #e.g /mnt/iDriveShare/OdetteR/Registration_and_contours/mask/'+Mask_'+str_pat_id+'.json'

    f = open(path_treatment_masks+'Mask_'+str_pat_id+'.json')
    data = json.load(f)
    f.close()
    mask = data['Mask']
    return mask


def get_dist_mask_body(mask,body):
    tree = KDTree(body.points)
    d_kdtree, idx = tree.query(mask.points)
   
    return np.max(d_kdtree),np.mean(d_kdtree),np.std(d_kdtree)

def get_equal_body_for_mask(body1,h,k,r):
    d1 = pv.PolyData(body1).connectivity(largest=True)

    bx = d1.points[:,0]
    by = d1.points[:,1]
    bz = d1.points[:,2]

    indexes = (bx-h)**2+(by-k)**2<=(r*0.8)**2

    bx2 = bx[indexes==True]
    by2 = by[indexes==True]
    bz2 = bz[indexes==True]

    points22 = list(zip(bx2,by2,bz2))
    d11 = pv.PolyData(points22)

    theta = np.linspace(0,2*np.pi,300)

    x = r*0.5 * np.cos( theta ) +h
    y = r*0.5 * np.sin( theta ) +k

    body_crop = [] #array points (cloud)
    zs = [i for i in d11.points[:,2]]
    zss = sorted(list(dict.fromkeys(zs)))
    for z in zss:
        ptosxy = []
        for p in d11.points:
            if z==p[2]:
                ptosxy.append([float(p[0]),float(p[1])])
        pp1= Polygon(ptosxy).buffer(0)
        pp2 = Polygon(list(zip(x,y)))
        pp3 = intersection(pp1,pp2)
        try:
            bx,by = pp3.exterior.coords.xy[0],pp3.exterior.coords.xy[1]
        except:
            coords = [[len(list(x.exterior.coords)),list(x.exterior.coords)] for x in pp3.geoms]
            bx,by = np.array((sorted(coords)[-1][-1]))[:,0],np.array((sorted(coords)[-1][-1]))[:,1]

        for j in range(0,len(bx)):
            body_crop.append((float(bx[j]),float(by[j]),float(z)))

    bbody = pv.PolyData(body_crop)
   
    return body

def get_mask_out(trim_mask,r,h,k):
    mask2 = pv.PolyData(trim_mask)
    bx = mask2.points[:,0]
    by = mask2.points[:,1]
    bz = mask2.points[:,2]

    indexes = (bx-h)**2+(by-k)**2<=(r*0.5)**2

    bx2 = bx[indexes==True]
    by2 = by[indexes==True]
    bz2 = bz[indexes==True]
    points22 = list(zip(bx2,by2,bz2))
    mask_final =pv.PolyData(points22)
    return mask_final

def get_area(body_slice,start_x2,start_y2,pixel_spacing2):
    
    px,py = get_mask_nifti(body_slice,start_x2,start_y2,pixel_spacing2)
    slicee = list(zip(px,py))
    ctr = np.array(slicee).reshape((-1,1,2)).astype(np.int32)
    mask = np.zeros((800,800), np.uint8)
    cv2.fillPoly(mask,pts=ctr,color=(255,255,255))
    dilated = dilation(mask2, disk(1))
    ret, thresh = cv2.threshold(dilated,0, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    mask2 = np.zeros((800,800), np.uint8)
    cv2.drawContours(mask2, cnt, 0, 255, -1)
    
    pixel_area_i=pixel_spacing2[0]*pixel_spacing2[1] #Get area of each pixel
    
    area_i_cm2 = (np.sum(mask2)/255)*pixel_area_i*0.01
    
    return area_i_cm2

def get_contour_submand(body,z_min):
    points_xy = []
    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])
    return points_xy


################
#  PTV RELATED METRICS FUNCTIONS
############
def get_index_farthest_point_from_cloud(points, cloud):
    tree = KDTree(cloud.points)
    dist_points, idxes_cloud = tree.query(points.points)
    max_d = max(dist_points)
    idx_point = np.where(dist_points == max_d)[0][0]
    idx_cloud = idxes_cloud[idx_point]
    return idx_point, idx_cloud

def get_index_nearest_point_from_cloud(points, cloud):
    tree = KDTree(cloud.points)
    dist_points, idxes_cloud = tree.query(points.points)
    min_d = min(dist_points)
    idx_point = np.where(dist_points == min_d)[0][0]
    idx_cloud = idxes_cloud[idx_point]
    return idx_point, idx_cloud

def get_distances_of_points_from_cloud(points, cloud):
    tree = KDTree(cloud.points)
    dist_points, idxes_cloud = tree.query(points.points)
    return dist_points

# For 3 given points in a circle, return the center (h,k) and radius r
def get_h_k_r(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    x12 = x1 - x2;
    x13 = x1 - x3;
    y12 = y1 - y2;
    y13 = y1 - y3;
    y31 = y3 - y1;
    y21 = y2 - y1;
    x31 = x3 - x1;
    x21 = x2 - x1;
 
    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2);
 
    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2);
 
    sx21 = pow(x2, 2) - pow(x1, 2);
    sy21 = pow(y2, 2) - pow(y1, 2);
 
    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))));
             
    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13))));
 
    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1);
 
    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g;
    k = -f;
    sqr_of_r = h * h + k * k - c;
 
    # r is the radius
    r = round(np.sqrt(sqr_of_r), 5);
    return [h, k, r]
    
def trim_posterior_PTV(cloud_PTV, contours_body, r_frac=1):    
    max_x = max(contours_body[:,0])
    min_x = min(contours_body[:,0])

    point0 = (min_x, get_max_y(min_x, contours_body))
    point1 = get_point_with_max_y_around_given_x(min_x/2, contours_body)
    point2 = get_point_with_max_y_around_given_x(0, contours_body)
    point3 = get_point_with_max_y_around_given_x(max_x/2, contours_body)
    point4 = (max_x, get_max_y(max_x, contours_body))

    h1,k1,r1 = get_h_k_r(point0, point1, point4)
    h2,k2,r2 = get_h_k_r(point0, point3, point4)
    # h3,k3,r3 = get_h_k_r(point0, point2, point4)
    h = np.mean([h1,h2])
    k = np.mean([k1,k2])
    r = np.mean([r1,r2])
    
    max_z = max(contours_body[:,2])
    min_z = min(contours_body[:,2])
    z = np.mean([min_z,max_z])
    spacing_z = get_contour_z_spacing(contours_body)
    height = (max_z - min_z) + 2*spacing_z

    bounding_cylinder = Cylinder(center=[h,k,z], direction=[0,0,1], radius=r*r_frac, height=height)
    cloud_PTV.compute_implicit_distance(bounding_cylinder, inplace=True)
    cloud_PTV_trim = cloud_PTV.threshold(0.0, scalars="implicit_distance", invert=True)
    
    return cloud_PTV_trim, h, k, r*r_frac

def get_distances_from_contours(contours_PTV, contours_body,IMG_RES, r_frac=RADIUS_FRAC, smooth_iter=0):
    # ================================================================================
    # trim body and PTV contours to have matching z limits
    contours_body, contours_PTV = trim_contours_to_match_z(contours_body, contours_PTV)
    
    # get pyvista objects of PTV and body
    cloud_PTV = pv.PolyData(contours_PTV)
    cloud_body = pv.PolyData(contours_body)
    
    # get body surface
    surface_body = get_surface_marching_cubes(contours_body,IMG_RES).smooth(n_iter=smooth_iter)
    
    # split PTV cloud into inner and outer
    inner_PTV, outer_PTV = split_cloud_by_surface(cloud_PTV, surface_body)
    xdiff, ydiff = get_bounding_box_dimensions(contours_body)
    
    if len(outer_PTV.points) > 0 and xdiff > ydiff:
        # trim PTV cloud that's protruding outside of FOV
        num_outer_points_ref = len(outer_PTV.points)
        cloud_PTV_trim, h, k, r = trim_posterior_PTV(cloud_PTV, contours_body, 1)
        # redefine outer_PTV and inner_PTV
        inner_PTV_new, outer_PTV_new = split_cloud_by_surface(cloud_PTV_trim, surface_body)
        num_outer_points_new = len(outer_PTV_new.points)
        
        # if something was trimmed, trim again using smaller bounding cylinder
        if num_outer_points_new < num_outer_points_ref:
            cloud_PTV_trim, h, k, r = trim_posterior_PTV(cloud_PTV, contours_body, r_frac)
            inner_PTV, outer_PTV = split_cloud_by_surface(cloud_PTV_trim, surface_body)
    # ================================================================================
        
    if len(outer_PTV.points) > 0:
        distances_outer = -1*get_distances_of_points_from_cloud(outer_PTV, cloud_body)
        distances_inner = get_distances_of_points_from_cloud(inner_PTV, cloud_body)
        distances = np.concatenate((distances_outer,distances_inner))
    else:
        distances = get_distances_of_points_from_cloud(inner_PTV, cloud_body)
    return distances

#CODE TO CALCULATE VOLUMES
# snippet from rtdsm's get_cubemarch_surface
def get_mask_grid(contours, img_res,max_away=400.):
    #STEP1: Get the min X,Y values to set the top corner of the slices
    Xmin,Ymin = np.nanmin(contours[:,0]),np.nanmin(contours[:,1])
    Zmin = np.nanmin(contours[:,2])
    CornerOrg = [Xmin - 2*img_res[0], Ymin - 2*img_res[1]]

    #STEP2: convert the XY values to index positions in a 3D array 
    contours[:,0] = np.round((contours[:,0]-CornerOrg[0])/img_res[0])
    contours[:,1] = np.round((contours[:,1]-CornerOrg[1])/img_res[1])

    #STEP3: Determine how many slices are needed to cover the full structure and make an empty grid
    uniqueSlices = np.unique(contours[:,2][~np.isnan(contours[:,2])])
    nSlices = len(uniqueSlices)
    GridMaxInd = np.nanmax(contours[:,:2])   #the max of the X and Y index values
    MaskGrid = np.zeros((int(nSlices),int(GridMaxInd +2),int(GridMaxInd+2))) #NOTE: using ZYX here
    
    #STEP4: Make a list of the indices where the slice number changes
    deltaslice = contours[:,2] - np.roll(contours[:,2],1)
    Slices = np.where(deltaslice != 0)[0] #indexes where a new polygon begins
    for i in range(len(Slices)):
        CurrentSlice = contours[Slices[i],2]
        sliceInd = np.where(uniqueSlices == CurrentSlice)[0]
        if np.isnan(CurrentSlice):
            continue
        #get the list of points for that polygon
        if i == len(Slices)-1:
            iPoints = contours[Slices[i]:,:2]
        else:
            iPoints = contours[Slices[i]:Slices[i+1],:2] #just need the X and Y points
        # split iPoints into cluster
        idx_breaks = [0]
        if len(iPoints) > 1:
            for i in range(len(iPoints)-1):
                if distance.euclidean(iPoints[i], iPoints[i+1]) < max_away:
                    continue
                else:
                    idx_breaks.append(i+1)
        for i in range(len(idx_breaks)): #Make a polygon mask from the points
            start = idx_breaks[i]
            if i == len(idx_breaks)-1:
                rr, cc = polygon(iPoints[start:,1], iPoints[start:,0])
            else:
                end = idx_breaks[i+1]
                rr, cc = polygon(iPoints[start:end,1], iPoints[start:end,0]) #r AKA row is Y, c AKA col is X
            MaskGrid[sliceInd,rr, cc] = 1
    return MaskGrid

def get_volume(contours, img_res, max_away=400.):
    volume_voxel = img_res[0] * img_res[1] * img_res[2]
    grid_mask = get_mask_grid(contours.copy(),img_res, max_away)
    volume = volume_voxel * np.sum(grid_mask)
    return volume

def get_volumes_from_contours(contours_PTV, contours_body,IMG_RES,r_frac=RADIUS_FRAC, smooth_iter=0):
    BIG_AWAY = 400
    SMALL_AWAY = 30
    # ================================================================================
    contours_body, contours_PTV = trim_contours_to_match_z(contours_body, contours_PTV)
    cloud_PTV = pv.PolyData(contours_PTV)
    cloud_body = pv.PolyData(contours_body)
    IMG_RES2 = [0.51119071245194, 0.51119071245194, 3]
    
    surface_body = get_surface_marching_cubes(contours_body,IMG_RES2).smooth(n_iter=smooth_iter)
    inner_PTV, outer_PTV = split_cloud_by_surface(cloud_PTV, surface_body)
    xdiff, ydiff = get_bounding_box_dimensions(contours_body)
    if len(outer_PTV.points) > 0 and xdiff > ydiff:
        num_outer_points_ref = len(outer_PTV.points)
        cloud_PTV_trim, h, k, r = trim_posterior_PTV(cloud_PTV, contours_body, 1)
        inner_PTV_new, outer_PTV_new = split_cloud_by_surface(cloud_PTV_trim, surface_body)
        num_outer_points_new = len(outer_PTV_new.points)
        if num_outer_points_new < num_outer_points_ref:
            cloud_PTV_trim, h, k, r = trim_posterior_PTV(cloud_PTV, contours_body, r_frac)
            inner_PTV, outer_PTV = split_cloud_by_surface(cloud_PTV_trim, surface_body)
    # ================================================================================
            cloud_PTV = cloud_PTV_trim
        
    img_res_body = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours_body)]
    img_res_PTV = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours_PTV)]
    
    vol_body = get_volume(contours_body, img_res_body, BIG_AWAY)
    vol_PTV = get_volume(cloud_PTV.points, img_res_PTV, BIG_AWAY)
    if len(outer_PTV.points) > 0:
        vol_PTV_outer = get_volume(outer_PTV.points, img_res_PTV, SMALL_AWAY)
        vol_PTV_inner = vol_PTV - vol_PTV_outer
    else:
        vol_PTV_outer = 0
        vol_PTV_inner = vol_PTV
    return vol_body, vol_PTV_inner, vol_PTV_outer

################################################################
# BODY RELATED METRIC
################################################################

def get_equal_body_fov(body1,h,k,r):
    d1 = pv.PolyData(body1).connectivity(largest=True)

    bx = d1.points[:,0]
    by = d1.points[:,1]
    bz = d1.points[:,2]

    indexes = (bx-h)**2+(by-k)**2<=(r*0.8)**2

    bx2 = bx[indexes==True]
    by2 = by[indexes==True]
    bz2 = bz[indexes==True]

    points22 = list(zip(bx2,by2,bz2))
    d11 =pv.PolyData(points22)

    theta = np.linspace(0,2*np.pi,300)

    x = r*0.5 * np.cos( theta ) +h
    y = r*0.5 * np.sin( theta ) +k

    body_crop = [] #array points (cloud)
    zs = [i for i in d11.points[:,2]]
    zss = sorted(list(dict.fromkeys(zs)))
    for z in zss:
        ptosxy = []
        for p in d11.points:
            if z==p[2]:
                ptosxy.append([float(p[0]),float(p[1])])
        pp1= Polygon(ptosxy).buffer(0)
        pp2 = Polygon(list(zip(x,y)))
        pp3 = intersection(pp1,pp2)
        try:
            bx,by = pp3.exterior.coords.xy[0],pp3.exterior.coords.xy[1]
        except:
            coords = [[len(list(x.exterior.coords)),list(x.exterior.coords)] for x in pp3.geoms]
            bx,by = np.array((sorted(coords)[-1][-1]))[:,0],np.array((sorted(coords)[-1][-1]))[:,1]

        for j in range(0,len(bx)):
            body_crop.append((float(bx[j]),float(by[j]),float(z)))

    bbody = pv.PolyData(body_crop)
    return bbody

def get_max_between_contours_by2Dv2(body1,body2):
    z_vals = np.array(list(set(body1[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    
    z_vals2 = np.array(list(set(body2[:,2])))
    z_vals2 = z_vals2[~(np.isnan(z_vals2))]
    sorted_z2 = np.array(sorted(z_vals2))
  
    slices_body1 = []
    for z in z_vals:
        ptosxy = []
        for p in body1:
            if z==p[2]:
                ptosxy.append([float(p[0]),float(p[1])])
        slices_body1.append(ptosxy)
        
    slices_body2 = []
    for z in z_vals2:
        ptosxy = []
        for p in body2:
            if z==p[2]:
                ptosxy.append([float(p[0]),float(p[1])])
        slices_body2.append(ptosxy)
    
    distances = []
    for k in range(0,len(slices_body2)):
        distance = directed_hausdorff(slices_body1[k], slices_body2[k])
        distances.append(distance[0])
    max_index = distances.index(np.max(distances))
    return np.max(distances),np.mean(distances),np.median(distances)

def get_chamfer_distance(body1,body2):
    chamfer = pcu.chamfer_distance(body1, body2)
    return chamfer

def get_max_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    return np.max(d_kdtree)


def get_volume_body_from_contours(contours_body,IMG_RES):
    # ================================================================================
    cloud_body = pv.PolyData(contours_body)
    img_res_body = [IMG_RES[0], IMG_RES[1], 3]

    vol_body = get_volume(contours_body, img_res_body, BIG_AWAY)

    return vol_body

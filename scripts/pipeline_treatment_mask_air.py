"""
Created on Jul 2024 
@author: Odette Rios-Ibacache & James Manalad

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
from scipy.stats import sem
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d
from datetime import date

import numpy as np
from pyvista import Cylinder
import alphashape
import pyvista as pv
import pydicom
import pandas as pd
import sympy as sym
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from skimage.draw import polygon
import random

import json
import csv
import gc, os
from shapely import Polygon, intersection
RADIUS_FRAC = 0.75

from ipywidgets import *

IMG_RES = [0.51119071245194, 0.51119071245194, 3]
      
def get_equal_bodyv2(body2,body1,z_max,z_min,h,k,r):
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
    s_body22 = get_surface_marching_cubes(bbody2)
    s_body2 = pv.PolyData(s_body22).connectivity(largest=True)

    s_body11 = get_surface_marching_cubes(bbody1)
    s_body1 = pv.PolyData(s_body11).connectivity(largest=True)
    return s_body1,s_body2,bbody1

def get_keys(name,patient):
        pat_h = []
    #    for patient in PATIENTS_ALL:\n",
        path_RS = get_path_RS(patient,PATH_SRC)
        keys_body = get_body_keys(path_RS)
        sorted_keys_body = sort_body_keys(keys_body)
        ROI_keys = get_ROI_keys(path_RS)
    #ROI_keys = get_ROI_keys(path_RS)\n",
        for key in ROI_keys:
            try:
                key2 = key.split('_')
            except:
                key2 = key.split('-')
            for k in key2:
               #try k.lower().split('')\n",
                #print(k)\n",
                try:
                    #k.split('~')\n",
                    for p in k.lower().split('~'):
                        if p==str(name):
                            #if patient not in pat_h:\n",
                            pat_h.append(key)
                except:
                    if k.lower()==str(name):
                        #if patient not in pat_h:\n",
                        pat_h.append(key)
        return pat_h

def get_keys_v2(name,patient,path_RS0):
        #PATH_K = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+CT+'/'
        pat_h = []
        
        keys_body = get_body_keys(path_RS0)
        sorted_keys_body = sort_body_keys(keys_body)
        ROI_keys = get_ROI_keys(path_RS0)
    
        for key in ROI_keys:
            try:
                key2 = key.split('_')
            except:
                key2 = key.split('-')
            for k in key2:
               #try k.lower().split('')\n",
                #print(k)\n",
                try:
                    #k.split('~')\n",
                    for p in k.lower().split('~'):
                        if p==str(name):
                            #if patient not in pat_h:\n",
                            pat_h.append(key)
                except:
                    if k.lower()==str(name):
                        #if patient not in pat_h:\n",
                        pat_h.append(key)
        return pat_h

def get_min_mandible_slice(s_body,mandible):

    m_m = min(mandible.points[:,2])
    roi_z = np.argmin(abs((s_body.points)[:,2] - (m_m)))
        #print(roi_z)
    m_b1 = (s_body.points)[:,2][roi_z]
    return m_b1

def get_point_with_max_y_around_given_xv2(x, points):
    target_x = x
    max_y = -1
    for point in points:
        current_x, current_y = point[0:2]
        if abs(current_x - x) < 2:
            if current_y > max_y:
                max_y = current_y
                target_x = current_x
    return (target_x, max_y)

def get_closest_xv2(x, points):
    closest_x = points[0][0]
    min_abs_diff = 10000
    for point in points:
        current_x = point[0]
        abs_diff = abs(current_x - x)
        if abs_diff < min_abs_diff:
            min_abs_diff = abs_diff
            closest_x = current_x
    return closest_x
    
#Estas  funciones son las mismas!
def get_max_yv2(x, points):
    target_x = get_closest_x(x, points)
    max_y = -1
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(target_x,1):
            if current_y > max_y:
                max_y = current_y
    return max_y
    
def get_x_max(y, points):
    max_x = -10000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_y,1) == round(y,1):
            if current_x>max_x:
                max_x = current_x
    return max_x

def get_y_max(x, points):
    max_y = -10000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(x,1):
            if current_y>max_y:
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
               


PATH_DEST = 'Thesis_120/air_mask'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS= ['volume_air']

def pipeline_mask(param_name='mask',path_k = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'):
    
    file = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_news = []
    
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_news.append(row[0])
    
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]
    t_init = process_time()

    for str_pat_id in ids_news[::-1]:
        #print('Processing patient ' + str_pat_id + '.') 
        patient_path = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'+str_pat_id
        
    
        # check if patient already has csv
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            print('Processing patient: ' + str_pat_id)
            t0 = process_time()
            
            contours = []
            key_bodies_to_save = []
            
            body_list = [d for d in os.listdir(patient_path) if d[0:4] == 'Body']
         
            f = open("/mnt/iDriveShare/OdetteR/Registration_and_contours/mask/Mask_"+str_pat_id+".json")
            data = json.load(f)
            f.close()
            mask = data['Mask']
           
            if len(body_list)==0:
               
                path_rs = get_path_RS(str_pat_id, path_k)
                print(path_rs)
                bodies_rs = get_body_keys(path_rs)
                bodies_sorted_rs = sort_body_keys(bodies_rs)
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
            
                    else:
                        if body not in key_bodies_to_save:
                            key_bodies_to_save.append(body)
                            contours.append(body_contour)
            else:
                    
                bodies = get_body_keys_not_RS(body_list)
                formatt = get_format(body_list)
        
                path_k1 = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
                CT,CBCTs = get_name_files(path_k1)
                path_CT = path_k1+CT
                path_rs_b0 = get_path_RS_v5(path_CT)
        
                bodies.insert(0,'BODY')
                
                for bodx in bodies:
                    if bodx=='BODY':
                    
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                        contours.append(body_contour)
                         
                    else:
                        for bodi in body_list:
                             body_in_folder = bodi.split('.')[0]
                             
                             if body_in_folder==bodx:
                                 formatt = bodi.split('.')[-1]
                             
                        path_RS0 = patient_path+'/'+bodx+'.'+formatt
                        print(path_RS0)
                        if formatt=='json':
                            f = open(path_RS0)
                            data = json.load(f)
                            f.close()
                            body_contour = np.array(data[bodx])
                            contours.append(body_contour)
                        else:
                            
                            body_contour = rtdsm.get_pointcloud(bodx, path_RS0, False)[0]
                            contours.append(body_contour)
                            
                key_bodies_to_save = bodies.copy()
            
            z_min,z_max = search_cuts_z(contours)           
            print(z_min,z_max)
            
            path_k1 = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            CT,CBCTs = get_name_files(path_k1)
            path_CT = path_k1+CT
            path_rs_b0 = get_path_RS_v5(path_CT)
            
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file) 

            # ================================================================================
            # CALCULATE PARAMETERS\
        
            print(key_bodies_to_save)
            r = get_info_fov(str_pat_id,key_bodies_to_save[1:])
            print(r)
            
            CT_path = get_info_replanned(str_pat_id,0)
                
            #pixel_spacing = [0.51119071245194, 0.51119071245194]
            pixel_spacing = get_start_position_dcm(CT_path)
            for key_body_n in range(0,len(key_bodies_to_save)):

                t1 = process_time()
                
                if key_bodies_to_save[key_body_n]=='BODY':
                    contour_body = contours[0]
                    contour_body_1 = contours[1]
                    
                    #h,k = get_center(contour_body_1,r) #Maybe hacer el FOV promedio? y aplicarlos a todos.
                    h,k = get_center2(path_k,str_pat_id)
                    body_sim = get_equal_body_for_mask(contour_body,h,k,r)
                    
                    trim_body2,trim_mask2 = trim_contours_to_match_zs(body_sim.points,pv.PolyData(mask).points,z_min,z_max)
                
                    d11 = get_mask_out(trim_mask2,r,h,k)
                    
                    
                    body_plus_mask = list(trim_body2.copy())
                    for j in pv.PolyData(d11).points:
                        body_plus_mask.append(j)
                    
                    vol_body = get_volume(pv.PolyData(trim_body2).points,[pixel_spacing[0],pixel_spacing[1], get_contour_z_spacing(trim_body2)])

                    vol_mask_plus_body = get_volume(pv.PolyData(body_plus_mask).points,[pixel_spacing[0],pixel_spacing[1], get_contour_z_spacing(trim_body2)])

                    volume = (vol_mask_plus_body-vol_body)*0.001
                   
                    
                    params = []
                    params.append(volume)
                 
                    
                    df[key_bodies_to_save[0]] = params
 
                else:
                   
                    
                    contour_body = contours[key_body_n]
                    h,k = get_center(contour_body,r)
                  
                    trim_body2,trim_mask2 = trim_contours_to_match_zs(contour_body,pv.PolyData(mask).points,z_min,z_max)
                   
                    d11 = get_mask_out(trim_mask2,r,h,k)
                    
                    body_plus_mask = list(trim_body2.copy())
                    for j in pv.PolyData(d11).points:
                        body_plus_mask.append(j)
                    #print(pixel_spacing+[3])
                    vol_body = get_volume(pv.PolyData(trim_body2).points,[pixel_spacing[0],pixel_spacing[1], get_contour_z_spacing(trim_body2)])

                    vol_mask_plus_body = get_volume(pv.PolyData(body_plus_mask).points,[pixel_spacing[0],pixel_spacing[1], get_contour_z_spacing(trim_body2)])
                    volume = (vol_mask_plus_body-vol_body)*0.001
                   
                    params = []
                    
                    params.append(volume)
                    
                    
                    df[key_bodies_to_save[key_body_n]] = params
                    
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
   
                # RECORD PARAMETERS
                # keep order same as ROWS!
            
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
    print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    pipeline_mask()

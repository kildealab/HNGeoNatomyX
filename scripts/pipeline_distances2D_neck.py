"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
from scipy.stats import sem
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
import numpy
import gc, os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d
from datetime import date
import numpy as np
import os
from pyvista import Cylinder
import point_cloud_utils as pcu
import pandas as pd
import pyvista as pv
import pydicom
from matplotlib import pyplot as plt
import scipy
from scipy.spatial.distance import directed_hausdorff
import circle_fit as cf
from shapely import Polygon, intersection
import math
import numpy as np
import os
from pyvista import Cylinder
import alphashape
import pandas as pd
import pyvista as pv
import pydicom
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import sympy as sym
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from skimage.draw import polygon
import random
import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
from scipy.stats import sem
from scipy.spatial import KDTree
from skimage.measure import marching_cubes

import gc, os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d

import numpy as np
import os
from pyvista import Cylinder

import pandas as pd
import pyvista as pv
import pydicom
from matplotlib import pyplot as plt

IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75

PATH_SRC =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'

from ipywidgets import *
def get_path_RS(pat_id, path_src):   #Obtiene el archivo del RT structure
    path_patient = os.path.join(path_src, pat_id)
    file_RS = [x for x in os.listdir(path_patient) if 'RS' in x][0]
    return os.path.join(path_patient, file_RS)
def get_body_keys(RS_file_path): #obtiene los keys de los body contours.
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'body' in x.lower()]
def sort_body_keys(keys_body): #Ordena los keys encontrados. de los body contours.
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

def get_patient_csv_filename(path_src, patient_num): #Obtiene el archivo csv file in the path?
    for fname in os.listdir(path_src):  
        if fname.split('_')[-1].split('.')[0] == str(patient_num):
            return fname
    print('No csv found for patient: ' + str(patient_num))
    return ''

def get_patient_data(path_src, patient_num):  #Obtiene los datos de los pacientes en el directorio? path_src
    #pero no hay ningún csv?
    patient_fname = get_patient_csv_filename(path_src, patient_num)
    path_csv = os.path.join(path_src, patient_fname)
    df = pd.read_csv(path_csv)
    return df

def get_param_value_dict_for_patient(path_src, patient_num, param_row_num):  
    df = get_patient_data(path_src, patient_num)
    return df.loc[param_row_num][1:]


#-----------------------------------------------
def sort_body_keys(keys_body): #Get body keys de la RT structure
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

def get_domain_from_keys(keys):
    xvals = []
    for key in keys:
        if 'body' not in key.lower():
            print('Key does not contain the word \'body\': ' + key)
        split_key = key.split('_')[-1].split('-')
        if len(split_key)==1 and 'body' in split_key[0].lower():
            xvals.append(0)
        if len(split_key)>1 and 'body' in split_key[0].lower():
            xvals.append(int(split_key[-1]))
    return np.array(xvals)

#----------------------------------------  
def get_patient_slope_for_num_fx(row_data, num_fx, fx_start=1):
    # print(row_data)
    idx_col = num_fx+1 # +1 because fx 1 has column index 2
    idx_start = fx_start+1
    row_data_trim = row_data[idx_start-1:idx_col+1].dropna() 
        # -1 because previous column is used to calculate first slope
        # +1 because last index specified is excluded
    # print(row_data_trim)
    xvals = get_domain_from_keys(row_data_trim.keys())
    print(xvals)
    yvals = row_data_trim.values
    print(yvals)
    # print(xvals, yvals)
    assert len(xvals) == len(yvals)
    if len(xvals) > 0:
        best_fit = LinearRegression().fit(xvals.reshape(-1, 1), yvals.reshape(-1, 1))
        slope = best_fit.coef_[0][0]
    else:
        slope = np.NAN
    return slope

#---------------------------------------
# column: body keys
# row: parameter
def get_param_df_for_patients(path_src, patient_list, param_name, param_row_num=0):
    data_dict = {'patient_num':[]}
    body_keys = []
    for patient_num in patient_list:
        dict_patient = get_param_value_dict_for_patient(path_src, patient_num, param_row_num)
        body_keys = list(set(body_keys + list(dict_patient.keys())))
    sorted_body_keys = sort_body_keys(body_keys)
    for body_key in sorted_body_keys:
        key = param_name + '_' + body_key
        data_dict[key] = []
    # print(data_dict)
    for patient_num in patient_list:
        data_dict['patient_num'].append(str(patient_num))
        dict_patient = get_param_value_dict_for_patient(path_src, patient_num, param_row_num)
        # print(dict_patient)
        for body_key in sorted_body_keys:
            key = param_name + '_' + body_key
            # print(body_key)
            if body_key in dict_patient.keys():
                data_dict[key].append(dict_patient[body_key])
            else:
                data_dict[key].append(np.nan)
    df = pd.DataFrame(data_dict)
    return df

def get_paramSlope_df_from_csv_individual(path_src, patient_list, param_name, param_row_num=0, fx_start=1):
    df_param = get_param_df_for_patients(path_src, patient_list, param_name, param_row_num)
    df_slope = get_paramSlope_df_from_param_df(df_param, fx_start)
    return df_slope

def get_paramSlope_df_from_csv_all(path_src_csv, patient_list, fx_start=1):
    df_param = pd.read_csv(path_src_csv)
    df_slope = get_paramSlope_df_from_param_df(df_param, fx_start)
    return df_slope

def get_paramSlope_df_from_param_df(df_param, fx_start=1):
    data_dict_slope = {'patient_num':[]}
    idx_fx_start = fx_start+1
    for column_name in df_param.keys()[idx_fx_start:]:
        key = column_name.split('_')[0]+'-slope_' + '_'.join(column_name.split('_')[1:])
        data_dict_slope[key] = []
    for idx_patient in range(len(df_param.values)):
        row_data = df_param.iloc[idx_patient]
        data_dict_slope['patient_num'].append(row_data['patient_num'])
        for i, key in enumerate(list(data_dict_slope.keys())[1:]): # key at index 0 is patient_num so we start at 1 
            num_fx = fx_start + i
            slope = get_patient_slope_for_num_fx(row_data, num_fx, fx_start)
            data_dict_slope[key].append(slope)
    df_slope = pd.DataFrame(data=data_dict_slope)
    return df_slope

def get_resultant_vector(vectors):
    vectors = np.array(vectors)
    resultant_vec = np.array([0.,0.,0.])
    for vector in vectors:
        resultant_vec += vector
    return resultant_vec

def asSpherical(xyz):    #Para convertir a coordenadas esfericas
    #takes list xyz (single coord)
    x, y, z = xyz
    r = np.sqrt(x*x + y*y + z*z)
    theta = sym.acos(z/r)#*180/ sym.pi #to degrees
    phi = sym.atan2(y,x)#*180/ sym.pi
    return r, theta.evalf(), phi.evalf()

def get_bounding_radius(points, centre):
    distances_bound = []
    for point in points:
        dist = distance.euclidean(point, centre)
        distances_bound.append(dist)
    r_bound = np.max(distances_bound)
    return r_bound


def get_path_RS(pat_id, path_src):   #Obtiene el archivo del RT structure
    path_patient = os.path.join(path_src, pat_id)
    file_RS = [x for x in os.listdir(path_patient) if 'RS' in x][0]
    return os.path.join(path_patient, file_RS)

def get_ROI_keys(RS_file_path):  #Obtiene el ROI image
    RS_file = pydicom.read_file(RS_file_path)
    contour_keys = RS_file.StructureSetROISequence
    return [str(x.ROIName) for x in contour_keys]

def get_body_keys(RS_file_path): #obtiene los keys de los body contours.
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'body' in x.lower()]

def get_PTV_keys(RS_file_path): #de los PTV
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'ptv' in x.lower()]


#------------------------

def sort_body_keys(keys_body): #Ordena los keys encontrados. de los body contours.
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

#Obtiene la resolución en el eje z. El spacing de las slice del CT
def get_contour_z_spacing(contours):
    z_vals = np.array(list(set(contours[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    diff_arr = sorted_z[:-1] - sorted_z[1:]
    #print(abs(np.mean(diff_arr)))
    return abs(np.mean(diff_arr))

#Cortar las secciones que tienen el rango maximo y minimo.
def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2

def get_surface_marching_cubes(contours):
    
    img_res = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours)]
  
    verts, faces, pvfaces = rtdsm.get_cubemarch_surface(contours.copy(), img_res)
  
    mesh = pv.PolyData(verts, faces=pvfaces)
    
    return mesh.extract_surface()

def get_surface_marching_cubes_clusters(contours):
    img_res = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours)]
    verts, faces, pvfaces = rtdsm.get_cubemarch_surface_clusters(contours.copy(), img_res)
    mesh = pv.PolyData(verts, faces=pvfaces)
    return mesh.extract_surface()

#VE que lo es que esta dentro y afuera de del body contour.
def split_cloud_by_surface(cloud, surface):
    cloud.compute_implicit_distance(surface, inplace=True)
    inner_cloud = cloud.threshold(0.0, scalars="implicit_distance", invert=True)
  
    outer_cloud = cloud.threshold(0.0, scalars="implicit_distance", invert=False)
   
    return inner_cloud, outer_cloud

# For a given set of contours and x value, return closest x values that is in one of the contour points
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

# For a given set of contours and x value, return the maximum y value
def get_max_y(x, contours):
    target_x = get_closest_x(x, contours)
    max_y = -1
    for point in contours:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(target_x,1):
            if current_y > max_y:
                max_y = current_y
    return max_y

# For a given set of contours and x value, return point with maximum y value
def get_point_with_max_y_around_given_x(x, contours):
    target_x = x
    max_y = -1
    for point in contours:
        current_x, current_y = point[0:2]
        if abs(current_x - x) < 2:
            if current_y > max_y:
                max_y = current_y
                target_x = current_x
    return (target_x, max_y)

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

#Maximos y minimos valores para encontrar el bounding boz de los contours.
def get_bounding_box_dimensions(contours):
    max_x = max(contours[:,0])
    min_x = min(contours[:,0])
    diff_x = max_x - min_x

    max_y = max(contours[:,1])
    min_y = min(contours[:,1])
    diff_y = max_y - min_y
    
    return [diff_x, diff_y]


def get_time_body_body(str_pad_id):
    time = []
    data = pd.read_csv('CT_CBCT_dates.csv')  
    #for str_pat_id in PATIENTS_ALL:
    dates = np.array(data[str_pad_id])
    new_dates = [element for element in dates if str(element) != "nan"]
    
    return new_dates,dates

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

def get_info_fov(patient,path_k = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'):
    path_patient = path_k+patient
    file_RS = [x for x in os.listdir(path_patient) if 'kV' in x][0]
    path2 = os.path.join(path_patient, file_RS)
    files = [x for x in os.listdir(path2) if 'CT' in x]
    files2 = []
    for j in files:
        files2.append(os.path.join(path2, j))
    dc_file = pydicom.read_file(files2[0])
    return dc_file.ReconstructionDiameter
    


def cut_surf_v4(body1,body2,cloud2,r):
    
    points = tuple(zip(body2.points[:,0],body2.points[:,1]))
    alpha_shape = alphashape.alphashape(points, alpha=1)
    
    h = max(alpha_shape.exterior.coords.xy[0]) - (((max(alpha_shape.exterior.coords.xy[0])) - (min(alpha_shape.exterior.coords.xy[0])))/2)
    k = (max(alpha_shape.exterior.coords.xy[1]))- r
    
    
    max_z = max(cloud2[:,2])
    min_z = min(cloud2[:,2])

    spacing_z = get_contour_z_spacing(body2.points)
    height = (max_z - min_z)  + 2*spacing_z
   
    z = np.mean([min_z,max_z])
        
    bounding_cylinder = Cylinder(center=[h,k,z], direction=[0,0,1], radius=r, height=height)
    bounding_cylinder2 = get_surface_marching_cubes(bounding_cylinder.points)
    
    trimmed = body1.boolean_intersection(bounding_cylinder2)
    return trimmed



def get_neck_region_v3(body1,body2,cloud2,r): #point clouds
    c2_1,c1_1 = trim_contours_to_match_z(body2,body1)
    c2 = get_surface_marching_cubes(c2_1)
    c1 = get_surface_marching_cubes(c1_1).smooth(n_iter=2)

    max_x = max(c2.points[:,0])
    min_x = min(c2.points[:,0])
    
    h = max_x - ((max_x - (min_x))/2)
    k = (max(c2.points[:,1]))- r
    
    max_z = max(c2_1[:,2])
    min_z = min(c2_1[:,2])

    spacing_z = get_contour_z_spacing(c2.points)
    height = (max_z - min_z)  + 2*spacing_z
   
    z = np.mean([min_z,max_z])
        
    bounding_cylinder = Cylinder(center=[h,k,z], direction=[0,0,1], radius=r, height=height)
    bounding_cylinder2 = get_surface_marching_cubes(bounding_cylinder.points)
    
    
    return c2,bounding_cylinder2,c1

def get_dist_vector(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    body1["distances"] = d_kdtree
    vectors = []
    for j in d_kdtree:
        idx_point = np.where(d_kdtree == j)[0][0]
        idx_cloud = idx[idx_point]
        point1 = body1.points[idx_point]
        point2 = body2.points[idx_cloud]
        vectors.append(point2 - point1)
    x = np.mean(np.array(vectors)[:,0])
    y = np.mean(np.array(vectors)[:,1])
    z = np.mean(np.array(vectors)[:,2])
    return x,y,z 
    
def get_min_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    idx_point = np.where(d_kdtree == np.min(d_kdtree))[0][0]
    idx_cloud = idx[idx_point]
    point1 = body1.points[idx_point]
    point2 = body2.points[idx_cloud]
    
    return np.min(d_kdtree),point1,point2

def get_max_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    return np.max(d_kdtree)

def get_mean_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    return np.mean(d_kdtree)

#def trim_contours_to_match_z2(contours_1, contours_2): # 1: body, 2: PTV
 #   spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
  #  max_z = max(contours_1[:,2]) - 3*spacing_z
   # min_z = min(contours_1[:,2]) + 3*spacing_z 
    
#    contours_2 = np.array([x for x in contours_2 if x[2] > min_z])
 #   contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])    
    
  #  return contours_1,contours_2

def trim_contours_to_match_zv3(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = z_max #- 3*spacing_z
    min_z = z_min #+ 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] > min_z])
    
    return contours_1, contours_2

def get_max_yv2(x, points):
    target_x = get_closest_x(x, points)
    min_y = 100000
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(target_x,1):
            if current_y < min_y:
                min_y = current_y
    return min_y
    
def get_neck_region_v6(body1,body2,cloud2,r):
    #bbody2,bbody1 = trim_contours_to_match_z2(body2.points, body1.points)
    #c2_2 = get_surface_marching_cubes(bbody2.points)
    #c2 = get_surface_marching_cubes(body2.points)
    
    bbody2,bbody1 = trim_contours_to_match_z2(body2.points, body1.points)
    c2_3 = get_surface_marching_cubes(bbody2)
    cc2 = pv.PolyData(c2_3).connectivity(largest=True)
    
    c11 = get_surface_marching_cubes(body1.points).smooth(n_iter=0)
    c1 = pv.PolyData(c11).connectivity(largest=True)

    max_x = max(c2_3.points[:,0])
    min_x = min(c2_3.points[:,0])
    
    h = max_x - ((max_x - (min_x))/2)
    k = get_max_yv2(max_x, c2_3.points)
    
    
    max_z = max(c1.points[:,2])
    min_z = min(c1.points[:,2])
    spacing_z = get_contour_z_spacing(c1.points)
    height = (max_z - min_z)  #+ 3*spacing_z
   
    z = np.mean([min_z,max_z])
    mesh = pv.CylinderStructured(center=[h,k,z], direction=[0,0,1], theta_resolution=50,z_resolution=80,radius=r*0.5, height=height)
    bounding_cylinder3 = get_surface_marching_cubes(mesh.points).smooth(n_iter=0)
    print('boolean')
    s_body1 = c1.copy().boolean_intersection(bounding_cylinder3.copy())
    
    max_z2 = max(c2_3.points[:,2])
    min_z2 = min(c2_3.points[:,2])
    spacing_z2 = get_contour_z_spacing(c1.points)
    height2 = (max_z2 - min_z2) + 2*spacing_z2
   
    z2 = np.mean([min_z2,max_z2])
    
    bounding_cylinder = Cylinder(center=[h,k,z2], direction=[0,0,1], radius=r*2, height=height2)
    print('threshold')
    s_body1.compute_implicit_distance(bounding_cylinder, inplace=True)
    cloud_trim = s_body1.threshold(0.0, scalars="implicit_distance", invert=True)
    return cloud_trim,cc2

def search_cuts_z(contours):
    z_maxs = []
    z_mins = []
    for j in contours:
        z_maxs.append(max(j[:,2]))
        z_mins.append(min(j[:,2]))
    return max(z_mins),min(z_maxs)

def trim_contours_to_match_zs(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)

    max_z = z_max #+ spacing_z
    min_z = z_min #- spacing_z

    contours_1 = np.array([x for x in contours_1 if x[2] <= max_z and x[2] >= min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] <= max_z and x[2] >= min_z])

    return contours_1, contours_2

def get_equal_body(body1,body2,z_min,z_max,r):
    dd11 = pv.PolyData(body1).connectivity(largest=True)

#    bbody2,bbody1 = trim_contours_to_match_z2(body2.points, body1.points)

    bbody2,bbody1 = trim_contours_to_match_zv3(body2.points, dd11.points,z_min,z_max)
    d2 = pv.PolyData(bbody2).connectivity(largest=True)
#d11 = get_surface_marching_cubes(d1.points)
    c2_3 = get_surface_marching_cubes(d2.points)
    cc2 = pv.PolyData(c2_3).connectivity(largest=True)

    d1 = pv.PolyData(bbody1).connectivity(largest=True)
    c11 = get_surface_marching_cubes(d1.points).smooth(n_iter=1)
#c11 = get_surface_marching_cubes(body1.points).smooth(n_iter=0)
    cc1 = pv.PolyData(c11).connectivity(largest=True)
    c1 = c11.copy()

    max_x = max(c2_3.points[:,0])
    min_x = min(c2_3.points[:,0])

    h = max_x - ((max_x - (min_x))/2)
    k = (max(c2_3.points[:,1]))- r*0.5

    max_z = max(c1.points[:,2])
    min_z = min(c1.points[:,2])
    spacing_z = get_contour_z_spacing(c1.points)
    height = (max_z - min_z)  #+ 3*spacing_z

    z = np.mean([min_z,max_z])
    mesh = pv.CylinderStructured(center=[h,k,z], direction=[0,0,1], theta_resolution=50,z_resolution=80,radius=r*0.5, height=height)
    bounding_cylinder3 = get_surface_marching_cubes(mesh.points).smooth(n_iter=1)

    s_body1 = cc1.copy().boolean_intersection(bounding_cylinder3.copy())

    max_z2 = max(c2_3.points[:,2])
    min_z2 = min(c2_3.points[:,2])
    spacing_z2 = get_contour_z_spacing(c1.points)
    height2 = (max_z2 - min_z2) + 2*spacing_z2

    z2 = np.mean([min_z2,max_z2])

    bounding_cylinder = Cylinder(center=[h,k,z2], direction=[0,0,1], radius=r*2, height=height2)

    s_body1.compute_implicit_distance(bounding_cylinder, inplace=True)
    cloud_trim = s_body1.threshold(0.0, scalars="implicit_distance", invert=True)

    return cloud_trim,c2_3

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

# For a given set of contours and x value, return the maximum y value
def get_max_yv2(x, points):
    target_x = get_closest_x(x, points)
    max_y = -1
    for point in points:
        current_x, current_y = point[0:2]
        if round(current_x,1) == round(target_x,1):
            if current_y > max_y:
                max_y = current_y
    return max_y

# For a given set of contours and x value, return point with maximum y value
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

def get_point_with_min_y_around_given_xv2(x, points):
    target_x = x
    min_y = 10000
    for point in points:
        current_x, current_y = point[0:2]
        if abs(current_x - x) < 2:
            if current_y < min_y:
                min_y = current_y
                target_x = current_x
    return (target_x, min_y)

def get_length_only_central(body,z_min,z_max):
    #tree = KDTree(body2.points)
    #d_kdtree, idx = tree.query(body1.points)
    #body1["distances"] = d_kdtree
    central_z = np.mean([z_min,z_max])
    #vectors2 = []
    #for j in d_kdtree:
    spacing = get_contour_z_spacing(body.points)
    points_xy = []
    
    
    for j in body.points:
        if j[2]==central_z:
            points_xy.append([j[0],j[1]])
    #print(points_xy)
            
    min_x = min(np.array(points_xy)[:,0])
    max_x = max(np.array(points_xy)[:,0])
    
    point1 = (min_x, get_max_yv2(min_x, points_xy))
    #point1 = get_point_with_min_y_around_given_xv2(min_x/2, points_xy)
    #point2 = get_point_with_max_y_around_given_xv2(min_x/2, points_xy)
    point3 = get_point_with_max_y_around_given_xv2(0,points_xy)
    #point4 = get_point_with_max_y_around_given_xv2(max_x/2, points_xy)
    point5 = (max_x, get_max_y(max_x, points_xy))
    #point5 = get_point_with_min_y_around_given_xv2(max_x/2, points_xy)
    point6 = (0,min(np.array(points_xy)[:,1]))
    
    x1, y1 = point1
    #x2, y2 = point2
    x3, y3 = point3
    #x4, y4 = point4
    x5, y5 = point5
    x6,y6 = (0,min(np.array(points_xy)[:,1]))
    
    lx = np.sqrt((x1-x5)**2+(y1-y5)**2)
    ly = np.sqrt((x3-x6)**2+(y3-y6)**2)
    
    points_xy2 = []
    for j in body.points:
        if j[2]==(central_z):
            points_xy2.append([j[0],j[1],j[2]])
   
    return lx,ly

def get_elongation_only_central(body,z_min,z_max):
    #tree = KDTree(body2.points)
    #d_kdtree, idx = tree.query(body1.points)
    #body1["distances"] = d_kdtree
    central_z = np.mean([z_min,z_max])
    #vectors2 = []
    #for j in d_kdtree:
    spacing = get_contour_z_spacing(body.points)
    points_xy = []
    
    
    for j in body.points:
        if j[2]==central_z:
            points_xy.append([j[0],j[1]])
    #print(points_xy)
            
    min_x = min(np.array(points_xy)[:,0])
    max_x = max(np.array(points_xy)[:,0])
    
    tree = KDTree(points_xy)
    x_central = np.mean([max(np.array(points_xy)[:,0]),min(np.array(points_xy)[:,0])])
    y_central = np.mean([max(np.array(points_xy)[:,1]),min(np.array(points_xy)[:,1])])
    
    d_kdtree, idx = tree.query([x_central,y_central])
    idx_point = np.where(d_kdtree == np.min(d_kdtree))[0][0]
    point1 = points_xy[idx]
    
    point11 = (min_x, get_max_yv2(min_x, points_xy))
    point5 = (max_x, get_max_yv2(max_x, points_xy))

    R_min = np.sqrt((point1[0]-x_central)**2+(point1[1]-y_central)**2)
    R_max = np.sqrt((x_central-point5[0])**2+(y_central-point5[1])**2)
            
    return R_max,R_min,R_max/R_min    



def trim_contours_to_match_zs_neck(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV

    spacing_z = get_contour_z_spacing(contours_1)

    max_z = z_max 
    min_z = z_min + 3*spacing_z

    contours_1 = np.array([x for x in contours_1 if x[2] <= max_z and x[2] >= min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] <= max_z and x[2] >= min_z])
    
    

    return contours_1, contours_2


def trim_contours_to_match_zs_neckv3(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV

    spacing_z = get_contour_z_spacing(contours_1)

    max_z = z_max 
    min_z = z_min 

    contours_1 = np.array([x for x in contours_1 if x[2] <= max_z and x[2] >= min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] <= max_z and x[2] >= min_z])
    
    

    return contours_1, contours_2


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
    
    print(central_z,central_x,central_y)
    print(pv.PolyData(body).center)

    dists = np.sqrt((np.array(body)[:,0]-central_x)**2+(np.array(body)[:,1]-central_y)**2+(np.array(body)[:,2]-central_z)**2)
    
    R_mean = (np.mean(dists))

    R_min = (np.min(dists))
    R_max = (np.max(dists))


    return R_max,R_min,R_max/R_min,R_mean
    
def change_z_coordinates(structure,z_value):
    #z_value = get_zmin_mandible(mandible,strcuture)
    new_structure = translation_z(structure,z_value)
    return new_structure

def translation_z(structure,z_value):
    matrix = np.array(([1,0,0,0],[0,1,0,0],[0,0,1,-z_value]))
    new = []
    for j in structure:
        new.append(list(j[0:3])+[1])
    new2 = []
    for j in new:
        new2.append(matrix@j)
    return new2
    
def get_zmin_mandible(mR,body1):
    zs_b = body1[:,2].copy()
    zs_b = zs_b[~(np.isnan(zs_b))]
    zs_m =  mR[:,2].copy()
    mR_min = min(zs_m)
    roi_zcR = np.argmin(abs(zs_b-mR_min)) 
    cenzcR2 =  zs_b[roi_zcR]
    return cenzcR2


def get_keys_v2(name,patient,path_RS0):
        #PATH_K = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+CT+'/'
        pat_h = []
        
      
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
        
def get_keys(name,patient):
        pat_h = []
    #    for patient in PATIENTS_ALL:\n",
        path_RS = get_path_RS(patient,PATH_SRC)
     
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
        

def get_body_keys_not_RS(file_list):
    body_keys = []
    for k in file_list:
        key = k.split('.')[0]
        body_keys.append(key)
    #body_keys
    sorted_keys = sort_body_keys(body_keys)
    return sorted_keys

def get_format(file_list):
    formatt = file_list[0].split('.')[1]    
    return formatt

def get_path_RS_v5(path_CT):   #Obti
    file_RS = [x for x in os.listdir(path_CT) if 'RS' in x][0]
    return os.path.join(path_CT, file_RS)

def get_path_RS(pat_id, path_src):   #Obtiene el archivo del RT structure
    path_patient = os.path.join(path_src, pat_id)
    file_RS = [x for x in os.listdir(path_patient) if 'RS' in x][0]
    return os.path.join(path_patient, file_RS)

def get_body_keys(RS_file_path): #obtiene los keys de los body contours.
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'body' in x.lower()]
    
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

def search_cuts_z(contours):
    z_maxs = []
    z_mins = []
    for j in contours:
        z_maxs.append(max(j[:,2]))
        z_mins.append(min(j[:,2]))
    return max(z_mins),min(z_maxs)


def get_point_with_max_y_around_given_x(x, contours):
    target_x = x
    max_y = -1
    for point in contours:
        current_x, current_y = point[0:2]
        if abs(current_x - x) < 2:
            if current_y > max_y:
                max_y = current_y
                target_x = current_x
    return (target_x, max_y)
    
def centers(x1, y1, x2, y2, r):
    q = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    xx = (r ** 2 - (q / 2) ** 2) ** 0.5 * (y1 - y2) / q
    yy = (r ** 2 - (q / 2) ** 2) ** 0.5 * (x2 - x1) / q
    
    return ((x3 + xx, y3 + yy))

def get_center(body2,r):

    #dd11 = pv.PolyData(body1).connectivity(largest=True)
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
    k4 = y4[1] - np.sqrt(np.abs((r*0.5)**2 - (min_x/2-h)**2))
   
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
    
    
def get_equal_body_fov(body1,h,k,r):
    #body_2 = pv.PolyData(body2).connectivity(largest=True)

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
    #bbody2,bbody1 = trim_contours_to_match_zs(body_2.points, bbody.points,z_min,z_max)
    #s_body22 = get_surface_marching_cubes(bbody2)
    #s_body2 = pv.PolyData(s_body22).connectivity(largest=True)

    #s_body11 = get_surface_marching_cubes(bbody1)
    #s_body1 = pv.PolyData(s_body11).connectivity(largest=True)
    return bbody#s_body1,s_body2

def get_max_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    return np.max(d_kdtree)

def get_max_between_contours_by2Dv2(body1,body2):
    #distances = []
    z_vals = np.array(list(set(body1[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    
    z_vals2 = np.array(list(set(body2[:,2])))
    z_vals2 = z_vals2[~(np.isnan(z_vals2))]
    sorted_z2 = np.array(sorted(z_vals2))
    
    #for q in range(0,len(sorted_z)):
     #   for m in range(0,len(sorted_z2)):
      #      if sorted_z[q]==sorted_z2[m]:
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
    
    #print(slices_body1)
    #print(slices_body2)
    distances = []
    distances2=[]
    for k in range(0,len(slices_body1)):
        #if k[0][2]==p[0][2]:
        distance = directed_hausdorff(slices_body1[k], slices_body2[k])
        distance2 = directed_hausdorff(slices_body2[k], slices_body1[k])
        distances.append(distance[0])
        distances2.append(distance2[0])
    max_index = distances.index(np.max(distances))
    max_index2 = distances2.index(np.max(distances2))
    
    return np.max(distances),max_index,np.max(distances2),max_index2

def get_hauss(body1,body2):
    hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(body1, body2)
    hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(body2, body1)
    
    return hausdorff_a_to_b,hausdorff_b_to_a

def get_chamber(body1,body2):
    chamber = pcu.chamfer_distance(body1, body2)
    return chamber

def get_z_out_fov(body,h,k,r,z_min):
    #z1 = sorted(list(set(body[:,2])))
    #z = sorted([x for x in z1 if not math.isnan(x)])
    #print(z)
    z1 = sorted(list(set(body[:,2])))
    z = sorted([x for x in z1 if not math.isnan(x)])
    print(z)
    
    body1,contours2 = trim_contours_to_match_zs_neckv3(body, body,z[0],0)
    #body1,contours2 = trim_contours_to_match_zs_neck(body, body,z_min,0)
    d1 = pv.PolyData(body1).connectivity(largest=True)
    bx = d1.points[:,0]
    by = d1.points[:,1]
    bz = d1.points[:,2]

    #indexes = (bx-h)**2+(by-k)**2<=(r*0.5*0.85)**2
    indexes2 = (bx-h)**2+(by-k)**2>=(r*0.5*0.90)**2
    bz2=bz[indexes2==True]
    return max(bz2)   
    
    
    
def get_max_dist_body(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    return np.max(d_kdtree)

def get_max_between_contours_by2Dv2(body1,body2):
    #distances = []
    z_vals = np.array(list(set(body1[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    
    z_vals2 = np.array(list(set(body2[:,2])))
    z_vals2 = z_vals2[~(np.isnan(z_vals2))]
    sorted_z2 = np.array(sorted(z_vals2))
    
    #for q in range(0,len(sorted_z)):
     #   for m in range(0,len(sorted_z2)):
      #      if sorted_z[q]==sorted_z2[m]:
    slices_body1 = []
    print(z_vals)
    print(z_vals2)
    
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
    
    #print(slices_body1)
    #print(slices_body2)
    distances = []
    distances2=[]
    for k in range(0,len(slices_body1)):
        #if k[0][2]==p[0][2]:
        distance = directed_hausdorff(slices_body1[k], slices_body2[k])
        distance2 = directed_hausdorff(slices_body2[k], slices_body1[k])
        distances.append(distance[0])
        distances2.append(distance2[0])
    max_index = distances.index(np.max(distances))
    max_index2 = distances2.index(np.max(distances2))
    
    return np.max(distances),np.mean(distances),np.median(distances)

def get_hauss(body1,body2):
    hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(body1, body2)
    hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(body2, body1)
    
    return hausdorff_a_to_b,hausdorff_b_to_a

def get_chamber(body1,body2):
    chamber = pcu.chamfer_distance(body1, body2)
    return chamber
    
    
def get_center2(path_k,str_pat_id):

    isos = rtdsm.get_pointcloud('AcqIsocenter', path_k+'/'+str_pat_id+'/iso.dcm', False)[0]
    h = isos[0][0]
    k = isos[0][1]

    return h,k 

import os
import csv
import json 

PATH_DEST = 'distances2Dneck_metrics/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['d_max2D_neck','d_mean2D_neck','d_median2D_neck']


def pipeline_params_body(param_name='distances2Dneck', path_k = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'):
    file = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_news = []
    
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_news.append(row[0])
    
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]
    t_init = process_time()
    
    for str_pat_id in ids_news:
        #print('Processing patient ' + str_pat_id + '.') 
        patient_path = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'+str_pat_id
    
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            print('Processing patient: ' + str_pat_id)
            t0 = process_time()
            
            contours = []
            key_bodies_to_save = []
            
            body_list = [d for d in os.listdir(patient_path) if d[0:4] == 'Body']
            
            
            if len(body_list)==0:
                path_rs = get_path_RS(str_pat_id, path_k)
                bodies_rs = get_body_keys(path_rs)
                bodies_sorted_rs = sort_body_keys(bodies_rs)
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
            
                    else:
                        if body not in key_bodies_to_save:
                            
                            gc.collect()
                            
                            mandiblee = get_keys('mandible',str_pat_id)
                            #mandiblee
                            mR= rtdsm.get_pointcloud(mandiblee[0], path_rs, False)[0]
                            z_val = get_zmin_mandible(mR,body_contour)
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                    
                            gc.collect()
                            key_bodies_to_save.append(body)
                            contours.append(new_body)
                # =============================
            else:
                    
                bodies = get_body_keys_not_RS(body_list)
                formatt = get_format(body_list)
        
                path_k1 = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
                CT,CBCTs = get_name_files(path_k1)
                path_CT = path_k1+CT
                path_rs_b0 = get_path_RS_v5(path_CT)
        
                bodies.insert(0,'BODY')
                mandiblee = get_keys_v2('mandible',str_pat_id,path_rs_b0)
                mR= rtdsm.get_pointcloud(mandiblee[0], path_rs_b0, False)[0]
              
                for bodx in bodies:
                    if bodx=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                    
                        z_val = get_zmin_mandible(mR,body_contour)
                       
                        new_body = change_z_coordinates(body_contour,z_val)
                        new_body = pv.PolyData(new_body).points
                    
                        gc.collect()
                        
                        contours.append(new_body)
                         
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
                            z_val = get_zmin_mandible(mR,body_contour)
                       
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                            
                            contours.append(new_body)
                            
                        else:
                            body_contour = rtdsm.get_pointcloud(bodx, path_RS0, False)[0]
                            
                            z_val = get_zmin_mandible(mR,body_contour)
                       
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                            
                            contours.append(new_body)
                            
                            
                key_bodies_to_save = bodies.copy()
        
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file)
             
            r = get_info_fov(str_pat_id)
            
            gc.collect()
            #print('--')
            z_min,z_max = search_cuts_z(contours)
            #print('despues del zs')
            gc.collect()
            if z_min>=-27:
                print(z_min)
                z_necks = z_min
               
            else:
              neck_zss = []
              for key_body_n in range(1,len(key_bodies_to_save)):
                  #t1 = process_time()
                  
                  body = contours[key_body_n]
                  gc.collect()
              
                  h,k = get_center(body,r)
                  min_z = get_z_out_fov(body,h,k,r,z_min)
                  neck_zss.append(min_z)
       
              print(neck_zss)
              print(min(neck_zss))
              if min(neck_zss)>=-27:
                  z_neck = -27
              
              else:
                  z_neck = min(neck_zss)
              
              gc.collect()
                 
            #contours00 = []
            contours_neck = []
                    
            print(z_neck)
            for key_body_n in range(0,len(key_bodies_to_save)):
                t1 = process_time()
                if key_bodies_to_save[key_body_n]=='BODY':
                
                    contour_body_1 = contours[1]
                    contour_body = contours[key_body_n]
                    #h,k = get_center(contour_body_1,r) #Maybe hacer el FOV promedio? y aplicarlos a todos.
                    h,k = get_center2(path_k,str_pat_id)
                    body_sim = get_equal_body_fov(contour_body,h,k,r)
                    gc.collect()
                    
                    trim_body,trim_body2 = trim_contours_to_match_zs_neck(body_sim.points,body_sim.points,z_neck,0)
                                
                    gc.collect()
                    contours_neck.append(trim_body)
            
                else:
                    contour_body = contours[key_body_n]
                    gc.collect()
                    
                    trim_body,trim_body2 = trim_contours_to_match_zs_neck(contour_body,contour_body,z_neck,0)
                    contours_neck.append(trim_body)           
            
            z_min2,z_max2 = search_cuts_z(contours_neck)
            for key_body_n in range(0,len(contours_neck)):
            
            
                contours00 = contours_neck[0]
                c_body = contours_neck[key_body_n]
                c_trim_body,c_trim_body2 = trim_contours_to_match_zs(c_body,contours00,z_min2,z_max2)
            
                params = []
               
                #Rmax,Rmin,El,Rmean = get_elongation3D(c_trim_body,c_trim_body2)
                
                gc.collect()
     
                d_max,d_mean,d_median = get_max_between_contours_by2Dv2(c_trim_body,c_trim_body2)
                
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
     
    
               
                        #hd,max_index,hd2,max_index2  = get_max_between_contours_by2Dv2(trim_body,trim_body2)
                gc.collect()
                   #     params.append(hd)
                    #    params.append(hd2)
                params.append(d_max)
                params.append(d_mean)
                params.append(d_median)        
                

                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    pipeline_params_body()


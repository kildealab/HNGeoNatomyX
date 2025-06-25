import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
from scipy.stats import sem
from scipy.spatial import KDTree
from skimage.measure import marching_cubes
import numpy as np
import gc, os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D, art3d
from datetime import date

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
import sys
from shapely import Polygon, intersection


"""Gets the RS path from the RT structure file.  pat_id  is the patient ID and path_src is the path of the folder"""
def get_path_RS(pat_id, path_src):   
    path_patient = os.path.join(path_src, pat_id)  """FORMAT: path_src + '/'+pat_id  """
    file_RS = [x for x in os.listdir(path_patient) if 'RS' in x][0]  """ finds the RS file with the name 'RS.######.dcm'"""
    return os.path.join(path_patient, file_RS)

"""Gets the body contour key with the labels 'body' in it"""
def get_body_keys(RS_file_path): 
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'body' in x.lower()]

"""Gets the ROI key labels available in the RT dicom file"""
def get_ROI_keys(RS_file_path):  #Obtiene el ROI image
    RS_file = pydicom.read_file(RS_file_path)
    contour_keys = RS_file.StructureSetROISequence
    return [str(x.ROIName) for x in contour_keys]

"""Gets the PTV label in the keys available in the RT dicom file"""
"""NOTE: the PTV commonly use is the PTV_All in the RT file"""

def get_PTV_keys(RS_file_path): #de los PTV
    ROI_keys = get_ROI_keys(RS_file_path)
    return [x for x in ROI_keys if 'ptv' in x.lower()]

"""Sorts the key from least to greatest body keys"""
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

def get_patient_csv_filename(path_src, patient_num):
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

#-------------------------------------

def get_contour_z_spacing(contours):
    z_vals = np.array(list(set(contours[:,2])))
    z_vals = z_vals[~(np.isnan(z_vals))]
    sorted_z = np.array(sorted(z_vals))
    diff_arr = sorted_z[:-1] - sorted_z[1:]
    #print(abs(np.mean(diff_arr)))
    return abs(np.mean(diff_arr))


def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2

def get_surface_marching_cubes(contours,IMG_RES):
    img_res = [IMG_RES[0], IMG_RES[1], get_contour_z_spacing(contours)]
    verts, faces, pvfaces = rtdsm.get_cubemarch_surface(contours.copy(), img_res)
    mesh = pv.PolyData(verts, faces=pvfaces)
    return mesh.extract_surface()

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

def get_bounding_box_dimensions(contours):
    max_x = max(contours[:,0])
    min_x = min(contours[:,0])
    diff_x = max_x - min_x
    max_y = max(contours[:,1])
    min_y = min(contours[:,1])
    diff_y = max_y - min_y
    
    return [diff_x, diff_y]
    
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

def get_info_fov(keys,path_RT_structure):
    path_patient = path_RT_structure
    file_RS = sorted([x for x in os.listdir(path_patient) if 'kV' in x])
    #print(sorted([x for x in os.listdir(path_patient) if 'kV' in x])[p])
    rs = []
    for p in range(0,len(keys)):
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

def get_center(body1,body2,r):
    dd11 = pv.PolyData(body1).connectivity(largest=True)
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
    
def get_equal_body(body2,body1,z_min,z_max,r,h,k):
    bbody2,bbody1 = trim_contours_to_match_zv3(body1.points,body2.points,z_min,z_max)
    #bbody2,bbody1 = trim_contours_to_match_z2(body2.points, body1.points)
    d2 = pv.PolyData(bbody2).connectivity(largest=True)

    c2_3 = get_surface_marching_cubes(d2.points).smooth(n_iter = 0)
    cc2 = pv.PolyData(c2_3).connectivity(largest=True)
    
    d1 = pv.PolyData(bbody1).connectivity(largest=True)
    c11 = get_surface_marching_cubes(d1.points).smooth(n_iter = 0)

    cc1 = pv.PolyData(c11).connectivity(largest=True)
    c1 = c11.copy()

    #ax_x = max(c2_3.points[:,0])
    #in_x = min(c2_3.points[:,0])
    
    #h = max_x - ((max_x - (min_x))/2)
    #k = (max(c2_3.points[:,1]))- r*0.5

    max_z = max(c1.points[:,2])
    min_z = min(c1.points[:,2])
    
    spacing_z = get_contour_z_spacing(c1.points)
    height = (max_z - min_z)  #+ 3*spacing_z
   
    z = np.mean([min_z,max_z])
    mesh = pv.CylinderStructured(center=[h,k,z], direction=[0,0,1], theta_resolution=50,z_resolution=80,radius=r*0.5, height=height)
    bounding_cylinder3 = get_surface_marching_cubes(mesh.points).smooth(n_iter = 0)
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
    
def get_elongation_only_central(body,z_m,z_m2,contour0):
    points_xy = []
    
    for j in body:
        if j[2]==z_m:
            points_xy.append([j[0],j[1]])
    #print(points_xy)


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
    
def trim_contours_to_match_zv3(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] > min_z])
    
    return contours_1, contours_2
    
def search_cuts_z(contours):
    z_maxs = []
    z_mins = []
    for j in contours:
        z_maxs.append(max(j[:,2]))
        z_mins.append(min(j[:,2]))
    return max(z_mins),min(z_maxs)
    
def trim_contours_to_match_zs(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2
    
    
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
 
    return bbody2,bbody1

def get_keys(name,patient):
        pat_h = []
        path_RS = get_path_RS(patient,PATH_SRC)
        keys_body = get_body_keys(path_RS)
        sorted_keys_body = sort_body_keys(keys_body)
        ROI_keys = get_ROI_keys(path_RS)

        for key in ROI_keys:
            try:
                key2 = key.split('_')
            except:
                key2 = key.split('-')
            for k in key2:
              
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
        pat_h = []
        ROI_keys = get_ROI_keys(path_RS0)
        for key in ROI_keys:
            try:
                key2 = key.split('_')
            except:
                key2 = key.split('-')
            for k in key2:
                try:
                    for p in k.lower().split('~'):
                        if p==str(name):
                            #if patient not in pat_h:\n",
                            pat_h.append(key)
                except:
                    if k.lower()==str(name):
                        #if patient not in pat_h:\n",
                        pat_h.append(key)
        return pat_h

    
def get_min_mandible_slice(body1,mR):
    zs_b = body1[:,2].copy()
    zs_b = zs_b[~(np.isnan(zs_b))]
    zs_m =  mR[:,2].copy()
    mR_min = min(zs_m)
    roi_zcR = np.argmin(abs(zs_b-mR_min)) 
    cenzcR2 =  zs_b[roi_zcR]
    return cenzcR2

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
    
def get_length_bottom(body,z_min):
 
    points_xy = []
    
    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])

            
    min_x = min(np.array(points_xy)[:,0])
    max_x = max(np.array(points_xy)[:,0])
    
    point1 = (min_x, get_max_yv2(min_x, points_xy))
    #point1 = get_point_with_min_y_around_given_xv2(min_x/2, points_xy)
    #point2 = get_point_with_max_y_around_given_xv2(min_x/2, points_xy)
    point3 = get_point_with_max_y_around_given_xv2(0,points_xy)
    #point4 = get_point_with_max_y_around_given_xv2(max_x/2, points_xy)
    point5 = (max_x, get_max_y(max_x, points_xy))
    point6 = (0,min(np.array(points_xy)[:,1]))
    
    x1, y1 = point1
    #x2, y2 = point2
    x3, y3 = point3
    #x4, y4 = point4
    x5, y5 = point5
    x6,y6 = (0,min(np.array(points_xy)[:,1]))
    
    # eqn of circle be a*x^2 + b*y^2 +  c*x*y + d*x + e*y = f = 0
    
    #plt.plot(np.array(points_xy)[:,0],np.array(points_xy)[:,1],'o')
    #plt.plot([x1,x5],[y1,y5],'g-')
    #plt.plot(x2,y2,'ro')
    #plt.plot([x3,x6],[y3,y6],'r-')
    #plt.plot(x4,y4,'ro')
    #plt.plot(x5,y5,'ro'
    #plt.plot(x6,y6,'ro')
    
    #a = np.array([[x1**2, y1**2,x1*y1,x1,y1,1], [x2**2, y2**2,x2*y2,x2,y2,1], [x3**2, y3**2,x3*y3,x3,y3,1], [x4**2, y4**2,x4*y4,x4,y4,1],[x5**2, y5**2,x5*y5,x5,y5,1],[x6**2, y6**2,x6*y6,x6,y6,1]])
    #b = np.array([0, 0,0,0,0,0])
    #x = np.linalg.solve(a, b)
    lx = np.sqrt((x1-x5)**2+(y1-y5)**2)
    ly = np.sqrt((x3-x6)**2+(y3-y6)**2)
    
    #points_xy2 = []
    #for j in body.points:
     #   if j[2]==(central_z):
      #      points_xy2.append([j[0],j[1],j[2]])
    
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter(body.points[:,0], body.points[:,1], body.points[:,2],alpha=0.002,color='lightgray',marker='*')
    #ax.scatter(np.array(points_xy)[:,0],np.array(points_xy)[:,1],[central_z for i in range(0,len(points_xy))],'ro')
    #ax.scatter(np.array(points_xy2)[:,0],np.array(points_xy2)[:,1],[central_z+10*spacing for i in range(0,len(points_xy2))],'ro')
    
    #plt.show()
    
    dx = abs(min_x-max_x)
    dy = abs(y3-y6)
            
            
    return lx,ly,dx,dy
    
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
    
    point1 = (min_x, get_max_yv2(min_x, points_xy))
    #point1 = get_point_with_min_y_around_given_xv2(min_x/2, points_xy)
    #point2 = get_point_with_max_y_around_given_xv2(min_x/2, points_xy)
    #point3 = get_point_with_max_y_around_given_xv2(0,points_xy)
    point3 = (get_x_min(max_y,points_xy),max_y)
    #point4 = get_point_with_max_y_around_given_xv2(max_x/2, points_xy)
    point5 = (max_x, get_max_y(max_x, points_xy))
    #point5 = get_point_with_min_y_around_given_xv2(max_x/2, points_xy)
    point6 = (get_x_min(min_y, points_xy),min_y)
    
    x1, y1 = point1
    #x2, y2 = point2
    x3, y3 = point3
    #x4, y4 = point4
    x5, y5 = point5
    x6,y6 = point6

    lx = np.sqrt((x1-x5)**2+(y1-y5)**2)
    ly = np.sqrt((x3-x6)**2+(y3-y6)**2)
            
    return lx,ly


def get_length_lx_planev2(body,z_min):
    points_xy = []


    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])


    value = 0

    values_lx = []
    for k in points_xy:
        x1,y1 = k[0],k[1]

        #value = 0
        for p in points_xy:
            x,y = p[0],p[1]
            if abs(y1-y)<2:
                lxx = np.sqrt((x - x1)**2)
                #if value<lxx:
                value = lxx
                p1x = x
                p1y = y
                p2y = y1
                p2x = x1
                values_lx.append([value,p1x,p1y,p2x,p2y])

    maxx = sorted(values_lx)[-1]
    return maxx[0]

def get_length_ly_planev2(body,z_min):
    points_xy = []


    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])

    value = 0

    values_ly = []
    for k in points_xy:
        x1,y1 = k[0],k[1]
        #value = 0
        for p in points_xy:
            x,y = p[0],p[1]
            if abs(x1-x)<2:
                lyy = np.sqrt((y - y1)**2)
                #if value<lyy:
                value = lyy
                p1x = x
                p1y = y
                p2y = y1
                p2x = x1
                values_ly.append([value,p1x,p1y,p2x,p2y])
    maxx = sorted(values_ly)[-1]
    return maxx[0]
    
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

def get_path_RS_v5(path_CT):  
    file_RS = [x for x in os.listdir(path_CT) if 'RS' in x][0]
    return os.path.join(path_CT, file_RS)

    
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

"""Gets the key for the mandible RT structure (contour)"""
def get_key_mandible(patient,path_RS0):
  
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
    
def get_center2(path_CBCT_images,str_pat_id):

    isos = rtdsm.get_pointcloud('AcqIsocenter', path_CBCT_images+'/'+str_pat_id+'/iso.dcm', False)[0]
    h = isos[0][0]
    k = isos[0][1]

    return h,k 
import json


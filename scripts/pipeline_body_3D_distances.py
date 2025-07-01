"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import gc, os
import pandas as pd

import helpers 
from helpers import get_path_RS, get_body_keys, sort_body_keys, get_keysall
from helpers import get_name_files, get_body_keys_not_RS, get_info_fov
from helpers import get_path_RS_CT,  search_cuts_z, get_center_fov
from helpers import trim_contours_to_match_zs


IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75

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



#Cortar las secciones que tienen el rango maximo y minimo.
def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2


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

#Maximos y minimos valores para encontrar el bounding boz de los contours.
def get_bounding_box_dimensions(contours):
    max_x = max(contours[:,0])
    min_x = min(contours[:,0])
    diff_x = max_x - min_x

    max_y = max(contours[:,1])
    min_y = min(contours[:,1])
    diff_y = max_y - min_y
    
    return [diff_x, diff_y]  
    
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
    x = np.mean((np.array(vectors)[:,0]))
    y = np.mean((np.array(vectors)[:,1]))
    z = np.mean((np.array(vectors)[:,2]))
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


def get_dist_vector_plane_xy(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    body1["distances"] = d_kdtree
    vectors2 = []
    for j in d_kdtree:
        idx_point = np.where(d_kdtree == j)[0][0]
        idx_cloud = idx[idx_point]
        point1 = body1.points[idx_point]
        point2 = body2.points[idx_cloud]

        if point2[2]==point1[2]:
            vector = (point2 - point1)

            rxy =np.sqrt((vector[0])**2+(vector[1])**2)

            vectors2.append(rxy)
    return np.mean(vectors2)

def get_dist_vector_plane(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    body1["distances"] = d_kdtree
    vectors = []
    for j in d_kdtree:
        idx_point = np.where(d_kdtree == j)[0][0]
        idx_cloud = idx[idx_point]
        point1 = body1.points[idx_point]
        point2 = body2.points[idx_cloud]
        if point2[2]==point1[2]:
            vectors.append(point2 - point1)
    x = np.mean(np.abs((np.array(vectors)[:,0])))
    y = np.mean(np.abs((np.array(vectors)[:,1])))
    z = np.mean(np.abs((np.array(vectors)[:,2])))
    return x,y,z


def get_center_vectors(body1,body2):
    tree = KDTree(body2.points)
    d_kdtree, idx = tree.query(body1.points)
    body1["distances"] = d_kdtree
    valuesx = []
    valuesy = []

    for j in d_kdtree:
        idx_point = np.where(d_kdtree == j)[0][0]
        idx_cloud = idx[idx_point]
        point1 = body1.points[idx_point]
        point2 = body2.points[idx_cloud]
        if point2[2]==point1[2]:
            r = np.sqrt(np.sum(point2**2))-np.sqrt(np.sum(point1**2))
            if r<0:
               valuex =  - np.abs(point2[0]-point1[0])
               valuey = - np.abs(point2[1]-point1[1])
               valuesx.append(valuex)
               valuesy.append(valuey)
            elif r>0:
                valuex = np.abs(point2[0]-point1[0])
                valuey = np.abs(point2[1]-point1[1])
                valuesx.append(valuex)
                valuesy.append(valuey)
    if np.sum(valuesx)==0 and np.sum(valuesy)==0:
        x = 0 
        y = 0
        xmin = 0
        ymin = 0
        xmed = 0
        ymed = 0
    else:
        x = np.mean((np.array(valuesx)))
        y = np.mean((np.array(valuesy)))
        xmin = np.min(np.array(valuesx))
        xmed = np.median(np.array(valuesx))
        ymed = np.median(np.array(valuesy))
#    xmax = np.max(np.array(v
        ymin = np.min(np.array(valuesy))
#    z = np.max(np.abs((np.array(vectors)[:,2])))
    return x,y,r,xmin,ymin,xmed,ymed

def get_elongation_only_central3D(body,contour0):
    iso = contour0.center
    dists = np.sqrt((np.array(body.points)[:,0]-iso[0])**2+(np.array(body.points)[:,1]-iso[1])**2+(np.array(body.points)[:,2]-iso[2])**2)
    R_mean = (np.mean(dists))

    R_min = (np.min(dists))
    R_max = (np.max(dists))
        
    return R_max,R_min,R_mean 

def get_elongation_only_central(body,z_m,z_m2,contour0):

    points_xy = []


    for j in body.points:
        if j[2]==z_m:
            points_xy.append([j[0],j[1]])
    #print(points_xy)


    points_xy_0 = []
    for j in contour0.points:
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
    s_body22 = get_surface_marching_cubes(bbody2)
    s_body2 = pv.PolyData(s_body22).connectivity(largest=True)

    s_body11 = get_surface_marching_cubes(bbody1)
    s_body1 = pv.PolyData(s_body11).connectivity(largest=True)
    return s_body1,s_body2

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
    
def get_length_bottom(body,z_min):
   
    points_xy = []
    
    
    for j in body.points:
        if j[2]==z_min:
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
    
def get_contour_submand(body,z_min):
    points_xy = []


    for j in body.points:
        if j[2]==z_min:
            points_xy.append([j[0],j[1]])
            
    return points_xy
  
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
      #  print(d) 
    positions = sorted(positions, key=lambda x: x[-1])
    start_z = positions[0][2]
    start_x = positions[0][0]
    start_y = positions[0][1]
    pixel_spacing = d.PixelSpacing
    
    return start_x, start_y, start_z, pixel_spacing 
    
    
def get_area(body_slice,start_x2,start_y2,pixel_spacing2):
    
    px,py = get_mask_nifti(body_slice,start_x2,start_y2,pixel_spacing2)
    slicee = list(zip(px,py))
    ctr = np.array(slicee).reshape((-1,1,2)).astype(np.int32)
    mask = np.zeros((800,800), np.uint8)
    cv2.fillPoly(mask,pts=ctr,color=(255,255,255))
    
    ret, thresh = cv2.threshold(mask,0, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    mask2 = np.zeros((800,800), np.uint8)
    cv2.drawContours(mask2, cnt, 0, 255, -1)
    
    pixel_area_i=pixel_spacing2[0]*pixel_spacing2[1] #Get area of each pixel
    
    area_i_cm2 = (np.sum(mask2)/255)*pixel_area_i*0.01
    #area_i = cv2.contourArea(ctr)
    #area_i_cm2=area_i*0.01 #Pass to cm2
    return area_i_cm2
    
    
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

        #if len(CT_list) != 2:
         #   raise Warning('More than 2 CT directories found. This code may not perform as expected, as it was made for exactly one replan (2 CTs)')
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



def trim_contours_to_match_zs(contours_1,contours_2,z_min,z_max): 
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max 
    min_z = z_min 
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    return contours_1,contours_2 

def search_cuts_z(contours):
    z_maxs = []
    z_mins = []
    for j in contours:
        z_maxs.append(max(j[:,2]))
        z_mins.append(min(j[:,2]))
    return max(z_mins),min(z_maxs)

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
    
def get_center2(path_k,str_pat_id):

    isos = rtdsm.get_pointcloud('AcqIsocenter', path_k+str_pat_id+'/iso.dcm', False)[0]
    h = isos[0][0]
    k = isos[0][1]

    return h,k

PATH_DEST = 'body3D_distances/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['dmax','chamfer']

def pipeline_area_body(param_name='body3D_distances',path_contours, CSV_patient_ids,path_CBCTs):
    t_init = process_time()
    #CSV_patient_ids = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_patients = []

    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_patients.append(str(row[0]))

    #CHECK IF THE PATIENT ALREADY HAS A CSV FILE IN THE DESTINATION FOLDER

    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]

    for str_pat_id in ids_patients:
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
            
            if len(body_list)==0:
                path_rs = get_path_RS(str_pat_id, path_contours)
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
                            
            
            # initialize dataframe and define output file name
            #PARA GUARDAR LOS DATOS, DEFINE LOS NOMBRES. 
            
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
            z_min,z_max = search_cuts_z(contours)

            key_mandible = get_keys_v2('mandible',str_pat_id,path_rs_b0)
            mandible1 = rtdsm.get_pointcloud(key_mandible[0],path_rs_b0,False)[0]
            
            mandible= get_surface_marching_cubes(mandible1)
            contour0 = get_surface_marching_cubes(contours[0])
            
            
            z_min,z_max = search_cuts_z(contours)

            r = get_info_fov(str_pat_id)
            
            for key_body in key_bodies_to_save:
        
                #contour_body = rtdsm.get_pointcloud(key_body, path_RS, False)[0]
                q = key_bodies_to_save.index(key_body)
                print('working on '+key_bodies_to_save[q])
                contour_body = contours[q]
                params=[]
                if len(contour_body) == 0:
                    print('\tSkipping ' + key_body + '. Body contours array is empty.')
                    continue
                else:
                   
                    if key_body=='BODY':
                        contour_body = contours[0]
                        contour_body2 = contours[1]
                        
                        #h,k = get_center(contour_body,contour_body2,r) #Maybe hacer el FOV promedio? y aplicarlos a todos.
                        h,k = get_center2(path_k,str_pat_id)
                        
                        body_sim = get_equal_body_fov(contour_body,h,k,r)
                        gc.collect()
                        
                        trim_body,trim_body2 = trim_contours_to_match_zs(body_sim.points,body_sim.points,z_min,z_max)            
                        d_max = get_max_dist_body(pv.PolyData(trim_body),pv.PolyData(trim_body2))
                        chamber = get_chamber(trim_body,trim_body2)
                        b1_b2,b2_b1 = get_hauss(trim_body,trim_body2)
                        #gc.collect()
                        #hd,max_index,hd2,max_index2 = get_max_between_contours_by2Dv2(trim_body,trim_body2)
                        gc.collect()
                        #print('\t' + key_body + 'done.')
                        
                        params.append(d_max)
                        params.append(chamber)
                        #params.append(b1_b2)
                        #params.append(b2_b1)
                   #     params.append(hd)
                    #    params.append(hd2)
                        
                        
                    else:
                           
                        contour_body_0 = contours[0]
                        contour_body_1 = contours[1]
                        
                        #h,k = get_center(contour_body_0,contour_body_1,r) #Maybe hacer el FOV promedio? y aplicarlos a todos.
                                                
                        h,k = get_center2(path_k,str_pat_id)
                        body_sim = get_equal_body_fov(contour_body_0,h,k,r)
                        gc.collect()
                        trim_body,trim_body2 = trim_contours_to_match_zs(body_sim.points,contour_body,z_min,z_max)            
                        #gc.collect()
                        d_max = get_max_dist_body(pv.PolyData(trim_body),pv.PolyData(trim_body2))
                        chamber = get_chamber(trim_body,trim_body2)
                        b1_b2,b2_b1 = get_hauss(trim_body,trim_body2)
                        
                        #hd,max_index,hd2,max_index2  = get_max_between_contours_by2Dv2(trim_body,trim_body2)
                        gc.collect()
                        print('\t' + key_body + 'done.')
                        params.append(d_max)
                        params.append(chamber)
                       # params.append(b1_b2)
#                        params.append(b2_b1)
                        #params.append(hd)
                        #params.append(hd2)
        
        
                    df[key_body] = params
                  
                    gc.collect()
                # ================================================================================
                    #df[key_body] = params
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
                
if __name__ == "__main__":
    pipeline_area_body()



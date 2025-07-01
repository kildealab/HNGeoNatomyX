"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import numpy as np
import gc, os, csv
import pandas as pd


#IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75

PATH_SRC =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'


#Cortar las secciones que tienen el rango maximo y minimo.
def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2

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
  
    return bbody#s_body1,s_body2

def get_z_out_fov(body,h,k,r,z_min):
    #z1 = sorted(list(set(body[:,2])))
    #z = sorted([x for x in z1 if not math.isnan(x)])
    #print(z)
    print(h,k)
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
    print(bz2)
    return max(bz2)  


PATH_DEST = 'elongation3Dneck_metrics/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['Rmax','Rmin','Rmax/Rmin','Rmean']


def pipeline_params_body(param_name='elongation3Dneck',path_contours,CSV_patients_ids,path_CBCTs):
    t_init = process_time()
    #CSV_patient_ids = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_patients = []
    
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_patients.append(row[0])
    
    #CHECK IF THE PATIENT ALREADY HAS A CSV FILE IN THE DESTINATION FOLDER
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]
    
    for str_pat_id in ids_patients:
        patient_contours_path = path_contours+str_pat_id    
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            print('Processing patient: ' + str_pat_id)
            t0 = process_time()
            
            contours = []
            key_bodies_to_save = []
            
            body_list = [d for d in os.listdir(patient_path) if d[0:4] == 'Body']
            # e.g. path_full_CBCT_id = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            path_full_CBCT_id = path_CBCTs+str_pat_id+'/'
         
            #GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
            CT,CBCTs = get_name_files(path_full_CBCT_id)
                
            #SET THE PATH OF THE CT 
            path_CT = path_full_CBCT_id+CT
            
            #SET THE PATH FOR THE RS FILE FOR THE FRACTION 0 (CT SIM IMAGE)
            path_rs_b0 = get_path_RS_CT(path_CT)
            mandible = get_key_mandible(path_rs_b0)
            mandible_cloud = rtdsm.get_pointcloud(mandible, path_rs_b0, False)[0]
                            
            if len(body_list)==0:
                path_rs = get_path_RS(str_pat_id, path_CBCT)
                bodies_rs = get_body_keys(path_rs)
                bodies_sorted_rs = sort_body_keys(bodies_rs)
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
            
                    else:
                        if body not in key_bodies_to_save:
                            gc.collect()
                            z_val = get_min_mandible_slice(body_contour,mandible_cloud)
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                            gc.collect()
                            key_bodies_to_save.append(body)
                            contours.append(new_body)
            else:
                #GETS KEYS THAT ARE NOT IN THE RS FILE WITH THE BODY (FRACTION 0/CT SIM) CONTOUR 
                bodies = get_body_keys_not_RS(body_list)
                bodies.insert(0,'BODY')
                body_list.insert(0,'BODY')

                for bodx in range(0,len(bodies)):
                    if bodx=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                        z_val = get_min_mandible_slice(body_contour,mandible_cloud)
                        new_body = change_z_coordinates(body_contour,z_val)
                        new_body = pv.PolyData(new_body).points
                        gc.collect()
                        contours.append(new_body)
                         
                    else:
                        body_in_folder = body_list[bodx].split('.')[0]
                        format_single_contour = body_list[bodx].split('.')[-1]
                        path_RS0 = patient_path+'/'+bodies[bodx]+'.'+format_single_contour

                        if format_single_contour=='json':
                            f = open(path_RS0)
                            data = json.load(f)
                            f.close()
                            body_contour = np.array(data[bodies[bodx]])
                            z_val = get_min_mandible_slice(body_contour,mandible_cloud)
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                            contours.append(new_body)
                            
                        else:
                            body_contour = rtdsm.get_pointcloud(bodies[bodx], path_RS0, False)[0]
                            z_val = get_min_mandible_slice(body_contour,mandible_cloud)
                            new_body = change_z_coordinates(body_contour,z_val)
                            new_body = pv.PolyData(new_body).points
                            contours.append(new_body)
                            
                key_bodies_to_save = bodies.copy()
        
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file)
             
            r = get_info_fov(str_pat_id)
            z_min,z_max = search_cuts_z(contours)            
            z_neck = get_z_bottom_neck(z_min,key_bodies_to_save,contours)
              
            gc.collect()
          
            contours_neck = []
        
            for key_body_n in range(0,len(key_bodies_to_save)):
                t1 = process_time()
                if key_bodies_to_save[key_body_n]=='BODY':
                
                    contour_body_1 = contours[1]
                    contour_body = contours[key_body_n]
                    h,k = get_center2(path_k,str_pat_id) #Maybe hacer el FOV promedio? y aplicarlos a todos.
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
               
                Rmax,Rmin,El,Rmean = get_elongation3D(c_trim_body,c_trim_body2)
                
                gc.collect()
            
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
     
                
                params.append(Rmax)
                params.append(Rmin)
                params.append(El)
                params.append(Rmean)            
                

                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    pipeline_params_body()


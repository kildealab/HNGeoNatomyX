"""
Created on Jul 2023 
@author: James Manalad

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import gc, os
import helpers
import pandas as pd

import helpers
from helpers import get_path_RS, get_body_keys, sort_body_keys, get_keysall
from helpers import get_name_files, get_body_keys_not_RS, get_info_replanned
from helpers import search_cuts_z, get_start_position_dcm, get_info_fov

IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75


#Cortar las secciones que tienen el rango maximo y minimo.
def trim_contours_to_match_z(contours_1, contours_2): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = max(contours_1[:,2]) - 3*spacing_z
    min_z = min(contours_1[:,2]) + 3*spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2


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

def trim_contours_to_match_zv3(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
    #print(spacing_z)
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] > min_z])
    
    return contours_1, contours_2
    
def trim_contours_to_match_zs(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2
    

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

PATH_DEST = 'volumesPTV/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['vol_PTV_inner', 'vol_PTV_outer','vol_PTV_inner_body_ratio','vol_PTV_outer_body_ratio']
BIG_AWAY = 400
SMALL_AWAY = 30

def pipeline_volumes(param_name='volumesPTV',path_contours,CSV_patients_ids,path_CBCTs):
    t_init = process_time()
    #CSV_patient_ids = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_patients = []

    #READ THE CSV IDS FILE
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_patients.append(str(row[0]))

    #CHECK IF THE PATIENT ALREADY HAS A CSV FILE IN THE DESTINATION FOLDER

    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]

    for str_pat_id in ids_patients:
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
        patient_contours_path = path_contours+str_pat_id     
     
        # check if patient already has csv
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            #GETS PROCESSING TIME FOR THE PIPELINE
            t0 = process_time()
            print('Processing patient: ' + str_pat_id)
            
            contours = []
            key_bodies_to_save = []
            ptv = []
         
            body_list = [d for d in os.listdir(patient_path) if d[0:4] == 'Body']

            # e.g. path_full_CBCT_id = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            path_full_CBCT_id = path_CBCTs+str_pat_id+'/'
         
            #GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
            CT,CBCTs = get_name_files(path_full_CBCT_id)
                
            #SET THE PATH OF THE CT 
            path_CT = path_full_CBCT_id+CT
         
            if len(body_list)==0:
                #GETS THE RT STRUCTURE FILE (RS FILE)
                path_rs = get_path_RS(str_pat_id, path_CBCT)
                bodies_rs = get_body_keys(path_rs)
                bodies_sorted_rs = sort_body_keys(bodies_rs)
             
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
                    else:
                        #CHECKS IF THE BODY CONTOUR AND LABEL IS ALREADY SAVED IN CASE THERE MORE THAN 2 CONTOURS WITH THE SAME NAME
                        if body not in key_bodies_to_save:
                            key_bodies_to_save.append(body)
                            contours.append(body_contour)
            else:
                #GETS KEYS THAT ARE NOT IN THE RS FILE    
                bodies = get_body_keys_not_RS(body_list)

                #SET THE PATH FOR THE RS FILE FOR THE FRACTION 0 (CT SIM IMAGE)
                path_rs_b0 = get_path_RS_CT(path_CT)
                bodies.insert(0,'BODY')
           
                for bodx in range(0,len(bodies)):
                    if bodies[bodx]=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                        contours.append(body_contour)
                    else:
                        body_in_folder = body_list[bodx].split('.')[0]
                        format_single_contour = body_list[bodx].split('.')[-1]
                        path_RS0 = patient_path+'/'+bodies[bodx]+'.'+format_single_contour

                        #CHECKS WHICH IS THE FORMAT THAT THE CONTOURS ARE SAVED: .json or .dcm
                        if format_single_contour=='json':
                            f = open(path_RS0)
                            data = json.load(f)
                            f.close()
                            body_contour = np.array(data[bodies[bodx]])
                            contours.append(body_contour)

                        else:
                            body_contour = rtdsm.get_pointcloud(bodies[bodx], path_RS0, False)[0]
                            contours.append(body_contour)
                            
                key_bodies_to_save = bodies.copy()

            #GETS THE KEY LABELS OF ALL THE STRUCTURES AVAILABLE OT GET THE PTV LABEL
            keys_all = get_keysall(str_pat_id,path_rs_CT)
            ptv_key = [i for i in keys_all if i=='z_PTV_ALL' or i=='z_PTVAll' or i=='z_PTV_All' or i=='PTV_ALL' or i=='PTV_All']
            contour_ptv = rtdsm.get_pointcloud(ptv_key[0], path_rs_CT, False)[0]
            ptv.append(contour_ptv)
  
            z_min,z_max = search_cuts_z(contours)           
            
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file) 

            # ================================================================================
            # CALCULATE PARAMETERS
         
            contours_PTV = ptv[0]
            #GETS THE PATH OF THE CT SIM FOLDER INDEX = 0 OR FRACTION  0
            CT_image_path = get_info_replanned(str_pat_id,0,path_CBCTs)

            #GETS PIXEL SPACING RESOLUTION OF THE CT SIM IMAGE, IN OUR CENTRE IS [0.51119071245194, 0.51119071245194, 3]
            #IF CHANGE NEEDS TO BE DONE, MODIFIED THE def get_volume_from_contours, AND USE
            #THE PIXEL SPACING FROM THE NEXT FUNCTION 
            pixel_spacing = get_start_position_dcm(CT_image_path)

            #GETS THE RADIUS OF THE FOV
            r = get_info_fov(str_pat_id)
         
            for key_body_n in range(0,len(key_bodies_to_save)):
                params = []
                body_contour = contours[key_body_n]
             
                gc.collect()
                t1 = process_time()
             
                #GETS THE METRICS
                vol_body, vol_PTV_inner, vol_PTV_outer = get_volumes_from_contours(contours_PTV, body_contour,pixel_spacing)
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
          
                # RECORD PARAMETERS
                # keep order same as ROWS!
             
                params = []
             
                params.append(vol_PTV_inner)
                params.append(vol_PTV_outer)
                params.append(vol_PTV_inner/vol_body)
                params.append(vol_PTV_outer/vol_body)
             
                # record calculated values under body key
                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
    print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    #e.g. paths
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    
    pipeline_volumes(path_contours,CSV_patients_ids,path_CBCTs)

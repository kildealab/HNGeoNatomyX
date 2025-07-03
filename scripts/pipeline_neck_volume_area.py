"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import gc, os, csv, json
import pandas as pd

import helpers 
from helpers import get_path_RS, get_body_keys, sort_body_keys, get_keysall
from helpers import get_name_files, get_body_keys_not_RS, get_info_fov, get_info_replanned
from helpers import get_path_RS_CT,  search_cuts_z, get_center_fov, get_z_bottom_neck
from helpers import get_start_position_dcm, trim_contours_to_match_zs_neck, get_equal_body_fov
from helpers import get_key_mandible, get_min_mandible_slice, change_z_coordinates
from helpers import get_volume_body_from_contours, get_surface_area, get_area_across_slices


PATH_DEST = 'neck_volume_area/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['volume_neck','surface_area_neck','compactness_neck','area2D_avg']

BIG_AWAY = 400
SMALL_AWAY = 30

def pipeline_params_volume_neck(param_name='neck_volume_area', path_contours,CSV_patients_ids,path_CBCTs):
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
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
        patient_contours_path = path_contours+str_pat_id 
        
        # check if patient already has csv
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            print('Processing patient: ' + str_pat_id)
            t0 = process_time()
            
            contours = []
            key_bodies_to_save = []
            
            body_list = [d for d in os.listdir(patient_contours_path) if d[0:4] == 'Body']

            # e.g. path_full_CBCT_id = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            path_full_CBCT_id = path_CBCTs+str_pat_id+'/'
         
            #GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
            CT,CBCTs = get_name_files(path_full_CBCT_id)
                
            #SET THE PATH OF THE CT 
            path_CT = path_full_CBCT_id+CT
            
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
                            gc.collect()
                            
                            mandible = get_key_mandible(path_RS0)                            
                            mandible_contour = rtdsm.get_pointcloud(mandible, path_rs, False)[0]
                            z_slice_mandible = get_min_mandible_slice(body_contour,mandible_contour)
                            
                            new_body = change_z_coordinates(body_contour,z_slice_mandible)
                            new_body = pv.PolyData(new_body).points
                    
                            gc.collect()
                            key_bodies_to_save.append(body)
                            contours.append(new_body)
                # =============================
            else:
                    
                #GETS KEYS THAT ARE NOT IN THE RS FILE    
                bodies = get_body_keys_not_RS(body_list)
                gc.collect()
                #SET THE PATH FOR THE RS FILE FOR THE FRACTION 0 (CT SIM IMAGE)
                path_rs_b0 = get_path_RS_CT(path_CT)
                bodies.insert(0,'BODY')
                body_list.insert(0,'BODY')

                key_mandible = get_key_mandible(str_pat_id,path_rs_b0)
                mandible_contour = rtdsm.get_pointcloud(key_mandible,path_rs_b0,False)[0]
                
                for bodx in range(0,len(bodies)):
                    if bodies[bodx]=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                        z_slice_mandible = get_min_mandible_slice(body_contour,mandible_contour)
                       
                        new_body = change_z_coordinates(body_contour,z_val)
                        new_body = pv.PolyData(new_body).points
                        gc.collect()
                        contours.append(new_body)
                         
                    else:
                        for body_in_list in body_list:
                            body_in_folder = body_in_list.split('.')[0] 
                            if bodies[bodx]==body_in_folder:
                                format_single_contour = body_list[bodx].split('.')[-1]
                                path_RS0 = patient_contours_path+'/'+bodies[bodx]+'.'+format_single_contour

                                #CHECKS WHICH IS THE FORMAT THAT THE CONTOURS ARE SAVED: .json or .dcm
                                if format_single_contour=='json':
                                    f = open(path_RS0)
                                    data = json.load(f)
                                    f.close()
                                    body_contour = np.array(data[bodies[bodx]])

                                    z_slice_mandible = get_min_mandible_slice(body_contour,mandible_contour)
                                    new_body = change_z_coordinates(body_contour,z_val)
                                    new_body = pv.PolyData(new_body).points
                                    
                                    contours.append(new_body)
                                else:
                                    body_contour = rtdsm.get_pointcloud(bodies[bodx], path_RS0, False)[0]
                                    z_slice_mandible = get_min_mandible_slice(body_contour,mandible_contour)
                       
                                    new_body = change_z_coordinates(body_contour,z_val)
                                    new_body = pv.PolyData(new_body).points
                            
                                    contours.append(new_body)
                            
                            
                key_bodies_to_save = bodies.copy()
        
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file)
             
            #NOTE THAT ALL THE CBCTs MUST HAVE THE SAME RECONSTRUCTION DIAMETER/RADIUS
            #IF NOT PLEASE USE THE MODIFIED VERSION get_info_fov_minimum
            #WHICH SEARCHES IN ALL THE CBCTs FILES THE MINIMUM RADIUS
            r = get_info_fov(path_full_CBCT_id)
            
            gc.collect()
            z_min,z_max = search_cuts_z(contours)
            
            z_bottom_neck = get_z_bottom_neck(z_min,key_bodies_to_save,contours)
            gc.collect()
                 
            contours_neck = []
                    
            for key_body_n in range(0,len(key_bodies_to_save)):
                t1 = process_time()
                if key_bodies_to_save[key_body_n]=='BODY':
                    contour_body_1 = contours[1]
                    contour_body = contours[key_body_n]
                    h_fov,k_fov = get_center_fov(path_contours,str_pat_id)
                    
                    body_sim = get_equal_body_fov(contour_body,h_fov,k_fov,r)
                    gc.collect()
                    
                    trim_body,trim_body2 = trim_contours_to_match_zs_neck(body_sim.points,body_sim.points,z_bottom_neck,0)
                                
                    gc.collect()
                    contours_neck.append(trim_body)
            
                else:
                    contour_body = contours[key_body_n]
                    gc.collect()
                    
                    trim_body,trim_body2 = trim_contours_to_match_zs_neck(contour_body,contour_body,z_bottom_neck,0)
                    contours_neck.append(trim_body)           
            
            z_min_necks,z_max_necks = search_cuts_z(contours_neck)
            CT_path = get_info_replanned(str_pat_id,0)
                
            start_x, start_y, start_z, pixel_spacing = get_start_position_dcm(CT_path)
            
            for key_body_n in range(0,len(contours_neck)):
                t1 = process_time()
                contours00 = contours_neck[0]
                neck_contour = contours_neck[key_body_n]
                params = []
               
                vol_neck = get_volume_body_from_contours(neck_contour,pixel_spacing)
                surface_area_neck  = get_surface_area(neck_contour,pixel_spacing)
                
                compactness = surface_area_neck**3/vol_neck**2
                mean_area = get_area_across_slices(key_body_n,str_pat_id,neck_contour)
                
                gc.collect()
            
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
     
                
                params.append(vol_neck)
                params.append(surface_area_neck)
                params.append(compactness)            
                params.append(mean_area)  

                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    #e.g. paths
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    pipeline_params_volume_neck(path_contours,CSV_patients_ids,path_CBCTs)
    


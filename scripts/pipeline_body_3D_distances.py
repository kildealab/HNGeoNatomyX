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
from helpers import get_path_RS_CT,  search_cuts_z, get_center_fov, get_equal_body_fov
from helpers import trim_contours_to_match_zs, get_chamfer_distance, get_max_dist_body

def trim_contours_to_match_zs(contours_1, contours_2,z_min,z_max): # 1: body, 2: PTV
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max - spacing_z
    min_z = z_min + spacing_z
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    
    return contours_1, contours_2


def trim_contours_to_match_zs(contours_1,contours_2,z_min,z_max): 
    spacing_z = get_contour_z_spacing(contours_1)
 
    max_z = z_max 
    min_z = z_min 
    
    contours_1 = np.array([x for x in contours_1 if x[2] < max_z and x[2] > min_z])
    contours_2 = np.array([x for x in contours_2 if x[2] < max_z and x[2] > min_z])
    return contours_1,contours_2 


PATH_DEST = 'body3D_distances/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['dmax','chamfer']

def pipeline_body3D_distances(param_name='body3D_distances',path_contours, CSV_patient_ids,path_CBCTs):
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
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
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
                gc.collect()
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
                    else:
                        if body not in key_bodies_to_save:
                            key_bodies_to_save.append(body)
                            contours.append(body_contour)
            else:
                #GETS KEYS THAT ARE NOT IN THE RS FILE    
                bodies = get_body_keys_not_RS(body_list)
                gc.collect()
                #SET THE PATH FOR THE RS FILE FOR THE FRACTION 0 (CT SIM IMAGE)
                path_rs_b0 = get_path_RS_CT(path_CT)
                bodies.insert(0,'BODY')
                body_list.insert(0,'BODY')
                
                for bodx in range(0,len(bodies)):
                    if bodies[bodx]=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_b0, False)[0]
                        contours.append(body_contour)
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
                                    contours.append(body_contour)
                                else:
                                    body_contour = rtdsm.get_pointcloud(bodies[bodx], path_RS0, False)[0]
                                    contours.append(body_contour)
                            
                key_bodies_to_save = bodies.copy()
                            
            # initialize dataframe and define output file name
            #PARA GUARDAR LOS DATOS, DEFINE LOS NOMBRES. 
        
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file) 
            
            # ================================================================================
            # CALCULATE PARAMETERS
        
            z_min,z_max = search_cuts_z(contours)
            contour_body0 = contours[0]
            r = get_info_fov(str_pat_id)
            
            for key_body in range(0,len(key_bodies_to_save)):
                print('Working on '+key_bodies_to_save[key_body])
                contour_body = contours[key_body]
                params=[]
                gc.collect()
                if len(contour_body) == 0:
                    print('\tSkipping ' + key_body + '. Body contours array is empty.')
                    continue
                else:
                    t1 = process_time()
                    if key_bodies_to_save[key_body]=='BODY':
                        contour_body = contours[0]                        
                        h,k = get_center_fov(path_contours,str_pat_id)
                        body_sim = get_equal_body_fov(contour_body,h,k,r)
                        gc.collect()
                        
                        trim_body,trim_body2 = trim_contours_to_match_zs(body_sim.points,body_sim.points,z_min,z_max)            
                        dmax = get_max_dist_body(pv.PolyData(trim_body),pv.PolyData(trim_body2))
                        chamfer = get_chamfer_distance(trim_body,trim_body2)
                        print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')
                        gc.collect()
                        params.append(dmax)
                        params.append(chamfer)
               
                    else:
                        h,k = get_center_fov(path_contours,str_pat_id)
                        body_sim = get_equal_body_fov(contour_body0,h,k,r)
                        gc.collect()
                        trim_body,trim_body2 = trim_contours_to_match_zs(body_sim.points,contour_body,z_min,z_max)            
                      
                        dmax = get_max_dist_body(pv.PolyData(trim_body),pv.PolyData(trim_body2))
                        chamfer = get_chamfer_distance(trim_body,trim_body2)
                        print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')

                        gc.collect()
                        params.append(dmax)
                        params.append(chamfer)

        
                    df[key_body] = params
                  
                    gc.collect()
                # ================================================================================
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
            gc.collect()
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
                
if __name__ == "__main__":
    #e.g. paths
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    pipeline_body3D_distances(path_contours,CSV_patients_ids,path_CBCTs)



"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import gc, os, csv
import helpers
import pandas as pd

import helpers 
from helpers import get_path_RS, get_body_keys, sort_body_keys, get_keysall
from helpers import get_name_files, get_body_keys_not_RS, trim_contours_to_match_zs
from helpers import get_path_RS_CT,  search_cuts_z, get_info_fov
from helpers import get_key_mandible, get_equal_body_fov, get_center_fov, get_start_position_dcm
from helpers import get_info_replanned, get_min_mandible_slice_from_surface, get_elongation_only_central


ROWS = ['Rmax','Rmin','Rmax/Rmin','Rmean']
PATH_DEST = 'elongationsubmand_metrics/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)

def pipeline_Rmin_Rmax(param_name='elongationsubmand',file_ids,path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'):

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
        
        # Checks if the patient already has a CSV file
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
          
        else:
            #GETS PROCESSING TIME FOR THE PIPELINE
            t0 = process_time()
            print('Processing patient: ' + str_pat_id)

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
                            key_bodies_to_save.append(body)
                            contours.append(body_contour)
                            
                key_mandible = get_key_mandible(str_pat_id,path_rs)
                mandible_contour = rtdsm.get_pointcloud(key_mandible,path_rs,False)[0]
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
                         for body_in_list in body_list:
                            body_in_folder = body_in_list.split('.')[0]  
                            if bodies[bodx]==body_in_folder:
                                format_single_contour = body_list[bodx].split('.')[-1]
                                path_RS0 = patient_contours_path+'/'+bodies[bodx]+'.'+format_single_contour

                                #CHECKS IF THE CONTOUR POINTS ARE IN JSON OR DCM
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
                key_mandible = get_key_mandible(str_pat_id,path_rs_b0)
               
                mandible_contour= rtdsm.get_pointcloud(key_mandible,path_rs_b0,False)[0] 
            
            # Initializes dataframe and defines output file name
    
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file) 

            # ================================================================================
            # CALCULATE METRICS

            z_min,z_max = search_cuts_z(contours)
            
            #NOTE THAT ALL THE CBCTs MUST HAVE THE SAME RECONSTRUCTION DIAMETER/RADIUS
            #IF NOT PLEASE USE THE MODIFIED VERSION get_info_fov_minimum
            #WHICH SEARCHES IN ALL THE CBCTs FILES THE MINIMUM RADIUS
            r = get_info_fov(path_full_CBCT_id)
            
            body_contour_0 = []
            for key_body in range(0,len(key_bodies_to_save)):
                
                params = []
                contour_body = contours[key_body]
                if len(contour_body) == 0:
                    print('\tSkipping ' + key_body + '. Body contours array is empty.')
                    continue
                else:
                    t1 = process_time()
                    if key_bodies_to_save[key_body]=='BODY':
                        contour_body = contours[0]
                             
                        h,k = get_center_fov(path_contours,str_pat_id)
                        body_sim = get_equal_body_fov(contour_body,h,k,r)
                        trim_body1,trim_body2 = trim_contours_to_match_zs(body_sim.points,body_sim.points,z_min,z_max)

                        body_contour_0.append(body_contour)
                        gc.collect()
        
                    
                    else:
                       
                        body1 = pv.PolyData(contour_body)                
                        trim_body1,trim_body2 = trim_contours_to_match_zs(body1.points,body1.points,z_min,z_max)

                        gc.collect()
       
                    CT_path = get_info_replanned(str_pat_id,0)
            
                    start_x, start_y, start_z, pixel_spacing = get_start_position_dcm(CT_path) 
                    s_body1 = get_surface_marching_cubes(trim_body2,pixel_spacing)
                    gc.collect()
                    
                    mandible_slice_body = get_min_mandible_slice_from_surface(s_body1.points,mandiblee)
                    contour0 = get_surface_marching_cubes(body_contour_0[0],pixel_spacing)
                    gc.collect()
                    
                    mandible_slice_0 = get_min_mandible_slice_from_surface(contour0.points,mandiblee)
                    Rmin,Rmax,El,Rmean = get_elongation_only_central(s_body1.points,mandible_slice_body,mandible_slice_0,contour0.points)

                
                    print('\tProcess time of parameters (' + key_bodies_to_save[key_body] + '): ' + str(process_time()-t1) + ' s')

      
                    params.append(Rmin)
                    params.append(Rmax)
                    params.append(El)
                    params.append(Rmean)
            
                # records calculated values under each body key (RT treatment fraction)
                df[key_bodies_to_save[key_body]] = params
                # ================================================================================
                
            # writes data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
            gc.collect()
            
        print('DONE! The metrics are saved in the CSV file')
        gc.collect()

if __name__ == "__main__":
    #e.g. paths
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    pipeline_Rmin_Rmax(path_contours,CSV_patients_ids,path_CBCTs)



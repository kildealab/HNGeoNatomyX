"""
Created on Jul 2024 
@author: Odette Rios-Ibacache 

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
import pandas as pd
from time import process_time
import gc, os, csv, json

import helpers 
from helpers import get_path_RS, get_body_keys, sort_body_keys
from helpers import get_key_mandible, get_body_keys_not_RS, get_format, get_name_files
from helpers import get_path_RS_CT,  search_cuts_z, get_info_fov, get_center_fov, get_CT_CBCT_equal_body
from helpers import get_min_mandible_slice, trim_contours_to_match_zs, get_contour_submand
from helpers import get_start_position_dcm, get_area, get_info_replanned


IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75
from ipywidgets import *

#DEFINING DESTINATION FOLDER
PATH_DEST = 'submand_area/'

#CREATE THE DESTINATION FOLDER
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
    
#METRIC NAME IN THE CSV FILE
ROWS = ['area']


def pipeline_area_body(param_name='submand_area',path_contours,CSV_patient_ids,path_CBCTs):
    t_init = process_time()
    
    #CSV_patient_ids = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_patients = []
    
    #READ THE CSV IDS FILE
    with open(CSV_patient_ids, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_patients.append(row[0])
            
    #CHECK IF THE PATIENT ALREADY HAS A CSV FILE IN THE DESTINATION FOLDER
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]

    #READ THE IDS AND CREATE THE CSV FILE TO SAVE 
    for str_pat_id in ids_patients:
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
        patient_contours_path = path_contours+str_pat_id

        # Check if the patient already has csv
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            #GETS PROCESSING TIME FOR THE PIPELINE
            t0 = process_time()
            
            print('Processing patient: ' + str_pat_id)
            contours = []
            key_bodies_to_save = []

            #GETS THE BODY KEYS FROM THE PATH FOLDER WITH THE CONTOURS
            body_list = [d for d in os.listdir(patient_contours_path) if d[0:4] == 'Body']

            # e.g. path_full_CBCT_id = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            path_full_CBCT_id = path_CBCTs+str_pat_id+'/'

            #if the body contours are saved in an RS file 
            # It reads the file and gets the body contours available
            if len(body_list)==0:
                path_rs = get_path_RS(str_pat_id, path_CBCT)
                bodies_rs = get_body_keys(path_rs)
                bodies_sorted_rs = sort_body_keys(bodies_rs)
                
                for body in bodies_sorted_rs:
                    body_contour = rtdsm.get_pointcloud(body, path_rs, False)[0]
                    #CHECK IF THE KEY LABEL HAS A CONTOUR ASSOCIATED
                    if len(body_contour)==0:
                        print('\tSkipping ' +body + '. Body contours array is empty.')
                    else:
                        #CHECKS IF THE BODY IS NOT IN THE BODIES CONTOURS TO USE 
                        if body not in key_bodies_to_save:
                            key_bodies_to_save.append(body)
                            contours.append(body_contour)                
            else:
                #GETS THE BODY KEYS FOR THE CONTOURS THAT ARE NOT SAVED IN THE RS FILE
                bodies = get_body_keys_not_RS(body_list)
                
                #CHECK FORMAT TYPE OF THE CONTOUR FILES
                format_contours = get_format(body_list)

                #GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
                CT,CBCTs = get_name_files(path_full_CBCT_id)
                
                #SET THE PATH OF THE CT 
                path_CT = path_full_CBCT_id+CT
                path_rs_CT = get_path_RS_CT(path_CT)
                bodies.insert(0,'BODY')

                gc.collect()
                for bodx in range(0,len(bodies)):
                    if bodies[bodx]=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_CT, False)[0]
                        contours.append(body_contour)
                    else:
                        body_in_folder = body_list[bodx].split('.')[0]
                        format_single_contour = body_list[bodx].split('.')[-1]

                        #SET RS PATH FOR THE BODY CONTOUR
                        path_RS0 = patient_path+'/'+bodx+'.'+format_single_contour
                        
                        if format_single_contour=='json':
                            f = open(path_RS0)
                            data = json.load(f)
                            f.close()
                            body_contour = np.array(data[bodies[bodx]])
                            contours.append(body_contour)
                        else:
                            body_contour = rtdsm.get_pointcloud(bodies[bodx], path_RS0, False)[0]
                            contours.append(body_contour)
        
            #GETS THE KEY LABEL FOR THE MANDIBLE CONTOUR            
            key_mandible = get_key_mandible(str_pat_id,path_rs)
            #GETS THE POINTCLOUD FOR THE MANDIBLE CONTOUR
            mandible = rtdsm.get_pointcloud(key_mandible,path_rs,False)[0]
        
            df = pd.DataFrame({param_name : ROWS})
            out_file = param_name + '_' + str_pat_id + '.csv'
            out_path = os.path.join(PATH_DEST,out_file) 
            
            # ================================================================================
            # CALCULATE PARAMETERS
            #SEARCH THE SAME Z RANGE FOR THE BODY CONTOURS
            z_min,z_max = search_cuts_z(contours)
          
            for key_body_n in range(0,len(key_bodies_to_save)):
                t1 = process_time()
                params = []
                
            #NOTE THAT ALL THE CBCTs MUST HAVE THE SAME RECONSTRUCTION DIAMETER/RADIUS
            #IF NOT PLEASE USE THE MODIFIED VERSION get_info_fov_minimum
            #WHICH SEARCHES IN ALL THE CBCTs FILES THE MINIMUM RADIUS
                
                r = get_info_fov(path_full_CBCT_id)
                
                if key_bodies_to_save[key_body_n]=='BODY':
                    #GETS THE BODY CONTOUR FOR FRACTION 0 (CT SIM)
                    b1 = contours[key_body_n]
                    #GETS THE NEXT BODY CONTOUR FX > 0
                    b2 = contours[1]

                    #DEFINES THE POINT CLOUDS FOR EACH BODY CONTOUR
                    body1 = pv.PolyData(b1)
                    body2 = pv.PolyData(b2)

                    #GETS THE FOV CENTER 
                    h,k = get_center_fov(path_CBCTs,str_pat_id)
                    
                    #BOOLEAN OPERATION TO GET THE SAME CT AND CBCT SHAPE DUE TO THE FOV
                    s_body1,s_body2 = get_CT_CBCT_equal_body(body2,body1,z_max,z_min,h,k,r)

                    gc.collect()
        
                else:
                    #GETS THE BODY CONTOUR FOR THE GIVEN RT FRACTION 
                    
                    b1 = contours[key_body_n]
                    #GETS THE BODY FOR THE FRACTION 1
                    b2 = contours[1]

                    #SETS THE POINT CLOUDS
                    body1 = pv.PolyData(b1)
                    body2 = pv.PolyData(b2)

                    #TRIM THE BODY CONTOURS TO HAVE THE SAME Z RANGE
                    b2_1,s_body1 = trim_contours_to_match_zs(body2.points,body1.points,z_min,z_max)
                    gc.collect()

                #GETS THE Z VALUE (SLICE VALUE) WHERE THE MANDIBLE IS 
                z_m = get_min_mandible_slice(pv.PolyData(s_body1),mandible)
                
                #GETS THE MANDIBLE CONTOUR
                ptosxy_mandible = get_contour_submand(s_body1,z_m)
                slice_mandible = np.array(ptosxy_mandible)[:,0:2]

                #GETS THE IMAGE PATH OF THE PATIENT 
                image_path = get_info_replanned(str_pat_id,key_body_n,path_CBCTs)

                #GETS THE PIXEL SPACING, AND THE POINTS IN THE SPACE: X,Y,Z
                start_x2, start_y2, start_z2, pixel_spacing2 = get_start_position_dcm(image_path) 
                
                #CALCULATES THE AREA FROM THE GIVEN MANDIBLE CONTOUR
                area = get_area(slice_mandible,start_x2,start_y2,pixel_spacing2)
                
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + str(process_time()-t1) + ' s')
  
                params.append(area)
                
                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to CSV
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'

    pipeline_area_body(path_contours,CSV_patient_ids,path_CBCTs)



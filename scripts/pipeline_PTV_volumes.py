"""
Created on Jul 2023 
@author: James Manalad & Odette Rios-Ibacache

"""

import sys
sys.path.append('/rtdsm')
import rtdsm
from time import process_time
import gc, os, csv, json
import pandas as pd

import helpers
from helpers import get_path_RS, get_body_keys, sort_body_keys, get_keysall
from helpers import get_name_files, get_body_keys_not_RS, get_info_replanned
from helpers import search_cuts_z, get_start_position_dcm, get_info_fov
from helpers import get_path_RS_CT, get_volumes_from_contours

#SET THE RESOLUTION OF THE VOXEL IN THE CBCT MEDICAL IMAGE 
#THIS RESOLUTION IS FOR CROPPING PUPORSES OF THE PTV OUTSIDE THE BODY
#ADJUST AT CONVENIENCE. THE VOLUME CALCULATION IS DONE WITH EACH IMAGE RESOLUTION
IMG_RES = [0.51119071245194, 0.51119071245194, 3]

RADIUS_FRAC = 0.75 #CHANGE THIS PARAMETER ONLY IF THE METRICS ARE NOT PROPERLY CALCULATED
#I.E., IF YOU GET AN ERROR. THIS RADIUS IS TO CALCULATE THE RELATIVE PTV CHANGES

PATH_DEST = 'volumesPTV/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
ROWS = ['vol_PTV_inner', 'vol_PTV_outer','vol_PTV_inner_body_ratio','vol_PTV_outer_body_ratio']

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
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
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
         
            body_list = [d for d in os.listdir(patient_contours_path) if d[0:4] == 'Body']

            # e.g. path_full_CBCT_id = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'+str_pat_id+'/'
            path_full_CBCT_id = path_CBCTs+str_pat_id+'/'
         
            #GETS THE NAME FILES OF EACH CT OR CBCT MEDICAL IMAGE
            CT,CBCTs = get_name_files(path_full_CBCT_id)
                
            #SET THE PATH OF THE CT 
            path_CT = path_full_CBCT_id+CT
         
            if len(body_list)==0:
                #GETS THE RT STRUCTURE FILE (RS FILE)
                path_rs = get_path_RS(str_pat_id, path_contours)
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

            #GETS THE KEY LABELS OF ALL THE STRUCTURES AVAILABLE OT GET THE PTV LABEL
            keys_all = get_keysall(str_pat_id,path_rs_CT)
            ptv_key = [i for i in keys_all if i=='z_PTV_ALL' or i=='z_PTVAll' or i=='z_PTV_All' or i=='PTV_ALL' or i=='PTV_All']
            contour_ptv = rtdsm.get_pointcloud(ptv_key[0], path_rs_CT, False)[0]
            ptv.append(contour_ptv)
                        
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

            #NOTE THAT ALL THE CBCTs MUST HAVE THE SAME RECONSTRUCTION DIAMETER/RADIUS
            #IF NOT PLEASE USE THE MODIFIED VERSION get_info_fov_minimum
            #WHICH SEARCHES IN ALL THE CBCTs FILES THE MINIMUM RADIUS
            r = get_info_fov(path_full_CBCT_id)
         
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
    path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/'
    CSV_patient_ids =  '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    path_CBCTs = '/mnt/iDriveShare/Kayla/CBCT_images/kayla_extracted/'
    
    pipeline_volumes(path_contours,CSV_patients_ids,path_CBCTs)

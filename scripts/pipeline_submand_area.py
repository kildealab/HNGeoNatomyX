"""
Created on Jul 2024 
@author: Odette Rios-Ibacache & James Manalad

"""

import helpers 
from helpers import get_path_RS, get_body_keys, sort_body_keys
from helpers import get_key_mandible, get_body_keys_not_RS, get_format, get_name_files
from helpers import get_path_RS_CT,  search_cuts_z
import time

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
    ids_news = []
    
    #READ THE CSV IDS FILE
    with open(CSV_patient_ids, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_news.append(row[0])
            
    #CHECK IF THE PATIENT ALREADY HAS A CSV FILE IN THE DESTINATION FOLDER
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]

    #READ THE IDS AND CREATE THE CSV FILE TO SAVE 
    for str_pat_id in ids_news:
        #e.g. path_contours = '/mnt/iDriveShare/OdetteR/Registration_and_contours'
        patient_path = path_contours+str_pat_id

        # Check if patient already has csv
        if str_pat_id in existing_patients:
            print('Patient already has csv:' + str_pat_id)
            continue
        else:
            t0 = process_time()
            
            print('Processing patient: ' + str_pat_id)
            contours = []
            key_bodies_to_save = []
            
            body_list = [d for d in os.listdir(patient_path) if d[0:4] == 'Body']

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
                
                for bodx in bodies:
                    if bodx=='BODY':
                        body_contour = rtdsm.get_pointcloud('BODY', path_rs_CT, False)[0]
                        contours.append(body_contour)
                    else:
                        for bodi in body_list:
                             body_in_folder = bodi.split('.')[0]
                             
                             if body_in_folder==bodx:
                                 format_single_contour = bodi.split('.')[-1]
                                 
                        path_RS0 = patient_path+'/'+bodx+'.'+format_contours 
                        
                        if format_contours=='json':
                            f = open(path_RS0)
                            data = json.load(f)
                            f.close()
                            body_contour = np.array(data[bodx])
                            contours.append(body_contour)
                        else:
                            body_contour = rtdsm.get_pointcloud(bodx, path_RS0, False)[0]
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
            z_min,z_max = search_cuts_z(contours)
            #GETS CONTOUR FOR FRACTION 0
            contour_BODY = contours[0]
            
            for key_body_n in range(0,len(key_bodies_to_save)):
                params = []
                        
                t1 = process_time()
      
                r = get_info_fov(str_pat_id)
                
                if key_bodies_to_save[key_body_n]=='BODY':
                    b1 = contours[key_body_n]
                    b2 = contours[1]
                    t1 = process_time()
                    #distances = get_distances_from_contours(contours_PTV, contours_body)
                    body1 = pv.PolyData(b1)
                    body2 = pv.PolyData(b2)
       
                    h,k = get_center(body1,body2,r)

                    s_body1,s_body2 = get_equal_bodyv2(body2,body1,z_max,z_min,h,k,r)

                    gc.collect()
        
                else:
                    b1 = contours[key_body_n]
                    b2 = contours[1]
                  
                    body1 = pv.PolyData(b1)
                    body2 = pv.PolyData(b2)


                    b2_1,b1_1 = trim_contours_to_match_zs(body2.points,body1.points,z_min,z_max)
                    s_body1 = b1_1
                    gc.collect()

                z_m = get_min_mandible_slice(pv.PolyData(s_body1),mandible)
                ptosxy = get_contour_submand(s_body1,z_m)
                
                slice1 = np.array(ptosxy)[:,0:2]
                
                CT_path = get_info_replanned(str_pat_id,key_body_n)
                
                start_x2, start_y2, start_z2, pixel_spacing2 = get_start_position_dcm(CT_path) 
                
                area = get_area(slice1,start_x2,start_y2,pixel_spacing2)
                
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
    pipeline_area_body()



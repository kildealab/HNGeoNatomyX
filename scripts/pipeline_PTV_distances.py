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


PATH_SRC = '/mnt/iDriveShare/OdetteR/Registration_and_contours/Contours/' # location of RT-struct files

IMG_RES = [0.51119071245194, 0.51119071245194, 3]
RADIUS_FRAC = 0.75


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


PATH_DEST = 'Thesis_120/xvalues/'
if not os.path.isdir(PATH_DEST):
    os.makedirs(PATH_DEST)
    
ROWS = ['xmin', 'xmax', 'xmed', 'xave', 'xstd']

def pipeline_dist_body(param_name='xvalues',path_contours, path ):
    
    file = '/mnt/iDriveShare/OdetteR/Registration_and_contours/IDS_News_Partial.csv'
    ids_news = []
    
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            ids_news.append(row[0])
    
    existing_patients = [csv_filename.split('_')[-1].split('.')[0] for csv_filename in os.listdir(PATH_DEST)]
    t_init = process_time()

    for str_pat_id in ['630']:
        #print('Processing patient ' + str_pat_id + '.') 
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
            
            ptv = []
            if len(body_list)==0:
                path_rs = get_path_RS(str_pat_id, path_k)
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
                keys_all = get_keysall(str_pat_id,path_rs)
            
                ptv_key = [i for i in keys_all if i=='z_PTV_ALL' or i=='z_PTVAll' or i=='z_PTV_All' or i=='PTV_ALL' or i=='PTV_All']
                
                contour_ptv = rtdsm.get_pointcloud(ptv_key[0], path_rs, False)[0]
                ptv.append(contour_ptv)
                
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
                    

                        #path_RS0 = patient_path+'/'+bodx+'.'+formatt
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
                keys_all = get_keysall(str_pat_id,path_rs_b0)
            
                ptv_key = [i for i in keys_all if i=='z_PTV_ALL' or i=='z_PTVAll' or i=='z_PTV_All' or i=='PTV_ALL' or i=='PTV_All']
                
                contour_ptv = rtdsm.get_pointcloud(ptv_key[0], path_rs_b0, False)[0]
                ptv.append(contour_ptv)
                print(ptv_key[0])
                            
            
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

    
            contours_PTV = ptv[0]
            #CT_path = get_info_replanned(str_pat_id,0)
                
            #start_x2, start_y2, start_z2, pixel_spacing2 = get_start_position_dcm(CT_path)
            pixel_spacing2 = [IMG_RES[0],IMG_RES[1]] 
            
            for key_body_n in range(0,len(key_bodies_to_save)):
                contours_body = contours[key_body_n]
                params = []
                
#                try:
                
                t1 = process_time()
                distances = get_distances_from_contours(contours_PTV, contours_body,pixel_spacing2)
                    
                print('\tProcess time of parameters (' + key_bodies_to_save[key_body_n] + '): ' + str(process_time()-t1) + ' s')

                # RECORD PARAMETERS
                # keep order same as ROWS!
                

                
                params.append(np.min(distances))
                params.append(np.max(distances))
                params.append(np.median(distances))
                params.append(np.mean(distances))
                params.append(np.std(distances))
                
         

                df[key_bodies_to_save[key_body_n]] = params
                # ================================================================================
                
            # write data to csv
            df.to_csv(out_path, index=False)
            print('\t' + param_name + ' printed to csv: ' + out_path)
            print('Elapsed time for patient: ' + str((process_time()-t0)/60) + ' min')
        #print('DONE! Elapsed time for pipeline: ' + str((process_time()-t_init)/3600) + ' hours')
    

if __name__ == "__main__":
    pipeline_dist_body()



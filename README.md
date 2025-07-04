# HN-GeoNatomyX tool
A Python script for calculating geometrical metrics from the 3D and 2D shape of Head and Neck Cancer patients taken from planning CTs or simulation CTs (CT sims) and Cone Beam CTs (CBCTs) taken during the radiotherapy treatment. This code has been tested on data exported from Eclipse versions 15 and 18. This repository also contains semi-automatic contouring tool for the treatment mask (immobilization device used during the radiotherapy treatment).

## Table of Contents
  *  [Authors](#Authors)
  *  [Motivation](#Motivation)
  *  [Features](#Features)
  *  [Dependencies](#Dependencies)
  *  [Installation](#Installation)
  *  [Usage](#Usage)
  *  [License](#License)
  *  [How to Cite](#HowtoCite)
  *  [References](#References)

## Authors
Odette Rios-Ibacache and James Manalad

Contact email: <a href="mailto:odette.riosibacache@mail.mcgill.ca">odette.riosibacache@mail.mcgill.ca</a>

Website:  [www.kildealab.com](https://www.kildealab.com) 
 
## Motivation
### Quantification of head and neck cancer patients’ anatomical changes: longitudinal analysis and prediction of radiotherapy replanning need

Head and neck (HN) cancer patients undergoing radiotherapy may experience significant anatomical changes due to weight loss and tumor shrinkage. These changes can impact the effectiveness of the initial treatment plan, potentially necessitating treatment replanning. However, ad hoc replanning requires additional clinical staff time, which can lead to suboptimal and stressful treatment planning. Furthermore, currently, there is no established method for determining the total amount of anatomical variation in the head and neck region to decide whether replanning is necessary. This research aimed to identify and create metrics based on patient anatomical structures that can describe the anatomical alterations that patients may experience throughout the treatment and influence decisions regarding treatment replanning. These parameters were used to develop a machine learning classification model to predict if patients would likely undergo replanning. Based on the 3D shape and 2D contours of structures, we defined 43 parameters. We developed **HN-GeoNatomyX**, an extraction pipeline to automatically calculate the defined 43 parameters over the course of the radiotherapy treatment.

## Features
### Semi-automatic Contouring Tool for the Treatment Mask
  ![Optional Text](/diagrams/Mask_treatment.jpg)
### Extraction Pipelines for the Created 43 Geometrical Metrics

Based on the 3D anatomy and 2D contours of the body and radiotherapy (RT) structures—including the Planning Target Volume (PTV) and mandible—we defined **43 continuous quantitative metrics** to characterize anatomical changes during RT delivery. These changes account for variations observed between the planning CT simulation (CT sim) and CBCT, as well as across sequential CBCTs. **The metrics were organized into six categories**, as outlined in the tables below.
#### Body-Related Metrics
<p align="center">
<img src="/diagrams/body_diagram.png" width="900">
</p>

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Body Volume  | $$V_{Body}$$ | HN volume enclosed by a body contour  |
| Chamfer distance (3D) | $$CD_{Body}$$ | Average distance between pair nearest neighbour points from the body contour of the CT sim to the body contour of the CBCTs|
| Haussdorff distance (3D) | $$HD_{Body}$$ | Maximum distance between the body contour of the CT sim and a body contour|
| Maximum 2D distance| $$D_{Body}$$ | The maximum 2D distance between a CBCT and the CT sim body contour calculated across each plane z|
| Median 2D distance| $$\tilde{D}_{Body}$$| The median 2D distance between a CBCT and the CT sim body contour calculated across each z plane | 
| Average 2D distance | $$\bar{D}_{Body}$$| Average distance between a CBCT and the CT sim body contour, across each plane z|

#### Treatment Mask-Related Metrics
<p align="center">
<img src="/diagrams/mask_diagram.png" width="700">
</p>

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Maximum distance from body contour to treatment mask  | $`max`${$`B_{mask}`$} | Maximum distance in 3D between the treatment mask structure and the external body contour of each patient|
| Average distance from body contour to treatment mask  | $$\bar{B}_{mask}$$ | Average value of the distribution of the 3D distances|
| Standard deviation of distances from body contour to treatment mask | $$\sigma_{B_{mask}}$$ | Corresponds to the standard deviation of the distribution of the distances|
| Air volume between body and treatment mask| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|

#### PTV-Related Metrics
<p align="center">
<img src="/diagrams/PTV_diagram.png" width="480">
</p>

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Minimum distance from PTV to body  | $$x_{min}$$  | Shortest distance between the PTV and the body contour, taken from a set of minimum distances between these two contours|
| Maximum distance from PTV to body  | $$x_{max}$$ | Takes into account the minimum distances from the PTV to the body contour, similar to $x_{min}$ but taking the maximum value instead|
| Average distance from PTV to body | $$x_{avg}$$ | Average value from a set of minimum distances from PTV to the body contour|
| Median distance from PTV to body| $$x_{med}$$ | Median value from a set of minimum distances from PTV to the body contour|
| Standard deviation of the distance between PTV to body| $$x_{std}$$ | This metric uses the standard deviation of the minimum distances between the PTV and the body contour |

<p align="center">
<img src="/diagrams/PTV_diagram_VO_VI.png" width="700">
</p>

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Volume PTV inner| $$VI_{PTV}$$ | This volume corresponds to the PTV that is inside the body region. The volume was calculated using the same procedure as the calculation of the body volume|
| Volume PTV outer| $$VO_{PTV}$$ | Volume that relies outside the body region|
| Volume PTV inner ratio| $VI_{PTV}:LV_{Body}$ | Volume that relies inside the enclosed body region, normalized by the body volume that limits the region of interest|
| Volume PTV outer radio| $VO_{PTV}:LV_{Body}$  | Volume that relies outside the enclosed body region, normalized by the body volume that limits the region of interest|

#### Mandible-Related Metrics

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Minimum distance from the mandible to body contour | $$M_{min}$$ | Defines the minimum distance encountered between the mandible structure and the body contour |
| Median distance from the mandible to contour body  | $$M_{med}$$ | Defines the median value from the distribution of the distances encountered between the mandible structure and the body contour|
| Average distance from the mandible to the body | $$M_{avg}$$ | Represents the average distance between the mandible and the external body contour from each medical image|
| Standard deviation of distances from the mandible to body contour| $$\sigma_{M}$$ | Corresponds to the standard deviation of the distribution of the distances between the mandible and the external body contours|

#### Neck-Related Metrics

| Metric  | Symbol | Definition |
| ------------- | ------------|-------------|
| Neck Volume  | $$V_{Neck}$$  | Volume enclosed by the neck region. The method to calculate the volume was the same as for the body volume|
| Chamber neck distance  | $$CD_{Neck}$$ | Average distance between the pair of nearest points between two neck regions. The distances were calculated between the CT sim neck and the CBCTs neck region point clouds|
| Haussdorff neck distance | $$HD_{Neck}$$ | Maximum distance between the pair of nearest points between two neck regions|
| Maximum 2D neck distance| $$D_{Neck}$$ | Maximum 2D distance between the pair of nearest points between two neck regions|
| Median 2D neck distance| $$\tilde{D}_{Neck}$$ | Median 2D distance between the pair of nearest points between two neck regions|
| Average 2D neck distance| $$\bar{D}_{Neck}$$ | Average 2D distance between the pair of nearest points between two neck regions|
| Minimum 3D radius| $$R^{3D}_{min}$$ | Minimum 3D radius that can be drawn from the 3D geometric center of the CT sim point cloud to the points in the contour mesh|
| Maximum 3D radius| $$R^{3D}_{max}$$ | Maximum 3D radius that can be drawn from the 3D geometric center of the CT sim point cloud to the points in the contour mesh|
| Average 3D radius| $$R^{3D}_{avg}$$ | Average of all the 3D radius that can be drawn from the 3D geometric center of the CT sim point cloud to the points in the contour mesh|
| Ratio between minimum and maximum 3D radius | $$\varphi^{3D}_{R}$$ | Ratio between the $`R^{3D}_{max}`$ and $`R^{3D}_{min}`$. This metric characterizes the symmetry of the mesh|
| Average cross-sectional neck area| $$A^{2D}_{avg}$$ | This parameter describes the total space covered the neck region|
| Surface area | $$SA_{Neck}$$ | This parameter describes the amount of space enclosing the outside of a neck region|
| Compactness| $$C_{Neck}$$ | Inspired by Bribiesca et al. (2008), the compactness describes the shape of the neck, representing the degree to which the neck is compact. It is calculated by the ratio between the enclosing surface area and the volume of a neck region ($$SA^3_{Neck}/V^2_{Neck}$$). This metric is dimensionless and minimized by a sphere|


#### Submandibular-Related Metrics
<p align="center">
<img src="/diagrams/submand_diagram.png" width="750">
</p>

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Area  | $$A_{sub}$$ | Total space covered by the submandibular body contour plane|
| Minimum 2D radius  | $$R_{min}$$ | Minimum 2D radius between the center of the CT sim submandibular body contour plane and a submandibular body contour plane |
| Maximum 2D radius| $$R_{max}$$ | Maximum 2D radius between the center of the CT sim submandibular body contour plane and a submandibular body contour plane|
| Average 2D radius| $$R_{avg}$$ | Average 2D radius that can be made between the center of the CT sim submandibular body contour plane and a submandibular body contour plane|
| Ratio between minimum and maximum 2D radius| $$\varphi^{2D}_{R}$$ | Calculated by the ratio between $$R_{max}$$ and $$R_{min}$$. This metric characterizes the symmetry of the submandibular plane contour|
| Maximum longitudinal chord | $l_{y}$ | Maximum distance that can be drawn from the outermost points of the submandibular body contour plane in a longitudinal direction |
| Maximum lateral chord | $l_{x}$ | Maximum distance that can be drawn from the outermost points of the submandibular body contour plane in a lateral direction |

## Dependencies
  *  [Python](https://www.python.org/downloads/)>=3.6
  *  [scipy](https://scipy.org/)>=1.6.0
  *  [skimage](https://scikit-image.org/)>=0.24.0
  *  [numpy](https://numpy.org/)>=1.19.5
  *  [pyvista](https://docs.pyvista.org/) >=0.31.3
  *  [alphashape](https://alphashape.readthedocs.io/en/latest/index.html) >=1.3.1
  *  [pydicom](https://pydicom.github.io/pydicom/stable/) >=3.0.1
  *  [sklearn](https://scikit-learn.org/stable/index.html)>=1.5.2
  *  [shapely](https://shapely.readthedocs.io/en/stable/)>=2.0.2
  *  [opencv-python](https://opencv.org/get-started/)>=4.11.0.86
  *  [json](https://docs.python.org/3/library/json.html)>=3.13.5

## Installation
By pulling the latest version from GitHub. Please note that your Python installation should be 3.6 or later. 
```
pip install git+https://github.com/kildealab/HN-GeoNatomyX.git
```

## Usage
### Data Format
Patient medical images (CT sims and CBCTs) were stored using the [dicoPATH](https://github.com/kildealab/dicoPATH) script. The data must be stored in the following format to streamline the use of the code. Each patient directory contains the CT simulation (planning medical image) and the CBCT images per treatment fraction.   Each folder  stores the associated DICOM files and RT structure (RS dicom) file.

<pre> /path/to/patient/directories/ 
├── 📁patient ID1
│   ├── 📁DATE_Planning_CT_name 
│        ├── 📄CT....dcm 
│        ├── 📄CT....dcm 
│        ├── ... 
│        └── 📄RS....dcm 
│   ├── 📁DATE_kV_CBCT_1a
│        ├── 📄CT....dcm 
│        ├── 📄CT....dcm 
│        ├── ... 
│        └── 📄RS....dcm 
|   ├── 📁DATE_kV_CBCT_3a
│   ├── ... 
...
</pre>

The pipelines also require contours to be saved in a DCM or JSON file format in a different folder. Each body contour should be saved with its label in the DCM and JSON. Our medical center's convention is to have the first body contour (based on the CT sim) with the name 'BODY' in the RT structure file, while for each CBCT contour, it is referred to as 'Body-fraction treatment' (e.g. Body-5, for treatment fraction 5). **They are stored with the same label in each patient's contours folder**.
The Acquisition Isocenter of each CBCT should also be registered and saved. The point in the RT structure file is saved as 'AcqIsocenter' (this can be modified in the corresponding reading function) and saved as 'iso.dcm' in the contours folder. 

<pre> /path/to/patient/contours directories/ 
├── 📁Contours
│   ├── 📁 patient ID1
│        ├── 📄BODY.dcm 
│        ├── 📄Body-1.dcm 
│        ├── 📄Body-2.dcm 
│        ├── ... 
│        ├── 📄Body-N.dcm 
│        └── 📄iso.dcm 
│   ├── 📁 patient ID2
│        ├── 📄BODY.json 
│        ├── 📄Body-1.json 
│        ├── 📄Body-5.json
│        ├── ... 
│        ├── 📄Body-N.json 
│        └── 📄iso.dcm 
│   ├── 📁 patient ID3
│        ├── 📄BODY.dcm
│        ├── 📄Body-1.json 
│        ├── 📄Body-4.json
│        ├── ... 
│        ├── 📄Body-N.dcm 
│        └── 📄iso.dcm 
│   ├── ... 
...
</pre>

Examples are included in Jupyter Notebook format to demonstrate how to run the scripts for contouring and extracting the metrics, as well as to provide context for the methodology used. 

### Treatment Mask Contouring 

<p align="center">
<img src="/diagrams/mask_contouring_example.png" width="900">
</p>

The treatment mask contouring script is called [treatment_mask_contouring.py](/scripts/treatment_mask_contouring.py). You can find an example on how to use it in [treatment_mak_contouring_example](/scripts/treatment_mask_contouring_example.ipynb). The final contour of the treatment mask is saved as .JSON file labeled as *'Mask_'+ID.json* (ID: patient identifier) in a folder called **masks**. 

<pre> /path/to/masks directories/ 
├── 📁 masks
│        ├── 📄Mask_ID1....json
│        ├── 📄Mask_ID2....json
│        ├── ... 
│        └── 📄Mask_IDN....json 
...
</pre>

### Geometrical Metrics Scripts
Each set of metrics has its own pipeline for each metric category. The output of each pipeline is a folder with the name of the metrics set and the metrics values saved in CSV files for each patient. 

<pre> /path/to/metrics/ 
<p align="left">
├── 📁 metrics
│        ├── <img src="/diagrams/csv_icon.png" width="20"> metric_patientID1.csv
│        ├── <img src="/diagrams/csv_icon.png" width="20"> metric_patientID2.csv
│        ├── ... 
│        └──<img src="/diagrams/csv_icon.png" width="20"> metric_patientIDN.csv 
... </p> 
</pre>

#### Body-Related Metrics Pipelines
| Script | Description |
| ------------- |-----------|
| [pipeline_body_volume.py](/scripts/pipeline_body_volume.py) | Calculates $V_{Body}$ in $mm^3$ and saves the results as CSV files labelled *'body_volume_'+ID* (ID: patient identifier) in a folder called **'body_volume'**| 
| [pipeline_body_3D_distances.py](/scripts/pipeline_body_3D_distances.py) | Calculates $CD_{Body}$ and $HD_{Body}$ in $mm$ and saves the results as CSV files labelled *'body3D_distances_'+ID* in a folder called **'body3D_distances'** |
| [pipeline_body_2D_distances.py](/scripts/pipeline_body_2D_distances.py) |  Calculates $D_{Body}$, $`\tilde{D}_{Body}`$, and $`\bar{D}_{Body}`$ in $mm$ and saves the results as CSV files labelled *'body2D_distances_'+ID* in a folder called **'body2D_distances'**  |

#### Treatment Mask-Related Metrics Pipelines
| Script | Description |
|-----------|-------------|
| [pipeline_treatment_mask_distances.py](/scripts/pipeline_treatment_mask_distances.py) | Computes $`max`${$`B_{mask}`$}, $`\bar{B}_{mask}`$, and $`\sigma_{B_{mask}}`$. The results are saved as CSV files labelled 'distancesMask_'+ID* in a folder called **'distancesMask'**|
| [pipeline_treatment_mask_air.py](/scripts/pipeline_treatment_mask_air.py) |  Computes $`V^{air}_{Body-to-mask}`$. The results are saved as CSV files labelled 'airMask_'+ID* in a folder called **'airMask'** | 

#### PTV-Related Metrics Pipelines
| Script | Description |
|-----------|-------------|
| [pipeline_PTV_distances.py](/scripts/pipeline_PTV_distances.py) |  Calculates $x_{min}$, $`x_{max}`$, $`x_{avg}`$, $`x_{med}`$, and $`x_{std}`$ in $mm$ and saves the results as CSV files labelled *'xvalues_'+ID* in a folder called **'xvalues'**  |  
| [pipeline_PTV_volumes.py](/scripts/pipeline_PTV_volumes.py) |  Calculates $VI_{PTV}$ and $`VO_{PTV}`$ in in $mm^3$, and  $`VI_{PTV}:LI_{Body}`$ and $`VO_{PTV}:LI_{Body}`$. It saves the results as CSV files labelled *'volumesPTV_'+ID* in a folder called **'volumesPTV'**   |  

##### Mandible-Related Metrics Pipeline

| Script | Description | 
|-----------|-------------|
| [pipeline_mandible_distances.py](/scripts/pipeline_mandible_distances.py) | Calculates the metrics $`M_{min}`$, $`M_{med}`$, $`M_{avg}`$, and $`M_{std}`$. The results are saved in a CSV file named *'mandible_'+ID* in a folder called **'mandible_metrics'** |

##### Neck-Related Metrics Pipelines

| Script | Description | 
|-----------|-------------|
| [pipeline_neck_volume_area.py](/scripts/pipeline_neck_volume_area.py) | Calculates the metrics $`V_{Neck}`$, $`A^{2D}_{avg}`$, $`SA_{Neck}`$, and $`C_{Neck}`$. The results are saved in a CSV file named *'neck_volume_area_'+ID* in a folder called **'neck_volume_area'** |
| [pipeline_distances3D_neck.py](/scripts/pipeline_distances3D_neck.py) |  Calculates the metrics $`CD_{Neck}`$, $`HD_{Neck}`$, $`M_{avg}`$, and $`M_{std}`$. The results are saved in a CSV file named *'distances3Dneck_'+ID* in a folder called **'distances3Dneck_metrics'**|
| [pipeline_distances2D_neck.py](/scripts/pipeline_distances2D_neck.py) | Computes $D_{Neck}$, $`\tilde{D}_{Neck}`$, and $`\bar{D}_{Neck}`$ in $mm$. The results are saved in a CSV file named *'distances2Dneck_'+ID* in a folder called **'distances2Dneck_metrics'** |
| [pipeline_RminRmax3D_neck.py](/scripts/pipeline_RminRmax3D_neck.py) |  Calculates the metrics $`R^{3D}_{min}`$, $`R^{3D}_{max}`$, $`R^{3D}_{avg}`$, and $`\varphi^{3D}_{R}`$. The results are saved in a CSV file named *'elongation3Dneck_'+ID* in a folder called **'elongation3Dneck_metrics'**|

##### Submandibular-Related Metrics Pipelines

| Script |  Description | 
|--------------|------------|
| [pipeline_submand_area.py](/scripts/pipeline_submand_area.py) | Calculates the metrics $`A_{sub}`$. The results are saved in a CSV file named *'submand_area_'+ID* in a folder called **'submand_area'** |
| [pipeline_submand_Rmin_Rmax.py](/scripts/pipeline_submand_Rmin_Rmax.py) | Calculates the metrics $`R_{min}`$, $`R_{max}`$, $`R_{avg}`$, and $`\varphi^{2D}_{R}`$. The results are saved in a CSV file named *'elongationsubmand_'+ID* in a folder called **'elong_submand_metrics'** |
| [pipeline_lx_ly_distances.py](/scripts/pipeline_lx_ly_distances.py) |  Computes the metrics $`l_{x}`$ and $`l_{y}`$. The results are saved in a CSV file named *'lx_ly_'+ID* in a folder called **'lx_ly_metrics'** |

## How to Cite 
If you publish any work using this package, please make sure you acknowledge us by citing the following paper: Rios-Ibacache O., Manalad J., O'Sullivan-Steben K., Poon E., at al. (2025). Quantification of head and neck cancer patients’ anatomical changes: prediction of radiotherapy replanning need.
## License 
This project is provided under the GNU General Public License version 3 (GPLv3) to preserve open-source access to any derivative works. See the LICENSE file for more information.
## References
* Barker, J. L., Jr, Garden, A. S., Ang, K. K., O'Daniel, J. C., Wang, H., Court, L. E., Morrison, W. H., Rosenthal, D. I., Chao, K. S., Tucker, S. L., Mohan, R., & Dong, L. (2004). Quantification of volumetric and   geometric changes occurring during fractionated radiotherapy for head-and-neck cancer using an integrated CT/linear accelerator system. International journal of radiation oncology, biology, physics, 59(4), 960–970. https://doi.org/10.1016/j.ijrobp.2003.12.024
* Figen, M., Çolpan Öksüz, D., Duman, E., Prestwich, R., Dyker, K., Cardale, K., Ramasamy, S., Murray, P., & Şen, M. (2020). Radiotherapy for Head and Neck Cancer: Evaluation of Triggered Adaptive Replanning in Routine Practice. Frontiers in oncology, 10, 579917. https://doi.org/10.3389/fonc.2020.579917
* van Beek, S., Jonker, M., Hamming-Vrieze, O., Al-Mamgani, A., Navran, A., Remeijer, P., & van de Kamer, J. B. (2019). Protocolised way to cope with anatomical changes in head & neck cancer during the course of radiotherapy. Technical innovations & patient support in radiation oncology, 12, 34–40. https://doi.org/10.1016/j.tipsro.2019.11.001
* Patrick, H. M., & Kildea, J. (2022). Technical note: rtdsm-An open-source software for radiotherapy dose-surface map generation and analysis. Medical physics, 49(11), 7327–7335. https://doi.org/10.1002/mp.15900


    

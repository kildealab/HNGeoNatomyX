# HN-GeoNatomyX tool
A Python script for calculating geometrical metrics from the 3D and 2D shape of Head and Neck Cancer patients taken from planning CTs or simulation CTs (CT sims) and Cone Beam CTs (CBCTs) taken during the radiotherapy treatment. This code has been tested on data exported from Eclipse versions 15 and 18. This repository also contains semi-automatic contouring tool for the treatment mask (immobilization device used during the radiotherapy treatment).

## Table of Contents
  *  [Authors](#Authors)
  *  [Motivation](#Motivation)
  *  [Features](#Features)
  *  [Dependencies](#Dependencies)
  *  [Installation](#Installation)
  *  [Usage and Examples](#UsageandExamples)
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
  ### Semi-automatic contouring tool for the treatment mask
  ![Optional Text](Mask_treatment.jpg)
  ### Code for extracting the 43 parameters stored in scripts

Based on the 3D anatomy and 2D contours of the body and radiotherapy (RT) structures—including the Planning Target Volume (PTV) and mandible—we defined **43 continuous quantitative metrics** to characterize anatomical changes during RT delivery. These changes account for variations observed between the planning CT simulation (CT sim) and CBCT, as well as across sequential CBCTs. **The metrics were organized into six categories**, outlined in the next Table.
#### Body-related metrics
| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Body Volume  | $$V_{Body}$$ | HN volume enclosed by a body contour  |
| Chamfer distance (3D) | $$CD_{Body}$$ | Average distance between pair nearest neighbour points from the body contour of the CT sim to the body contour of the CBCTs|
| Haussdorff distance (3D) | $$HD_{Body}$$ | Maximum distance between the body contour of the CT sim and a body contour|
| Maximum 2D distance| $$D_{Body}$$ | The maximum 2D distance between a CBCT and the CT sim body contour calculated across each plane z|
| Median 2D distance| $$\tilde{D}_{Body}$$| The median 2D distance between a CBCT and the CT sim body contour calculated across each z plane | 
| Average 2D distance | $$\bar{D}_{Body}$$| Average distance between a CBCT and the CT sim body contour, across each plane z|

#### Treatment mask-related metrics
| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Maximum distance from body contour to treatment mask  | $$max$$ { $$B_{mask}$$ } | Maximum distance in 3D between the treatment mask structure and the external body contour of each patient|
| Average distance from body contour to treatment mask  | $$\bar{B}_{mask}$$ | Considers the average value of the distribution of the 3D distances|
| Standard deviation of distances from body contour to treatment mask | $$\sigma_{B_{mask}}$$ | Corresponds to the standard deviation of the distribution of the distances|
| Air volume between body and treatment mask| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|

#### PTV-related metrics
#### Mandible-related metrics

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Minimum distance from the mandible to body contour | $$M_{min}$$ | Defines the minimum distance encountered between the mandible structure and the body contour |
| Median distance from the mandible to contour body  | $$M_{med}$$ | Defines the median value from the distribution of the distances encountered between the mandible structure and the body contour|
| Average distance from the mandible to the body | $$M_{avg}$$ | Represents the average distance between the mandible and the external body contour from each medical image|
| Standard deviation of distances from the mandible to body contour| $$\sigma_{M}$$ | Corresponds to the standard deviation of the distribution of the distances between the mandible and the external body contours|

#### Neck-related metrics

| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Neck Volume  | $$max$$ { $$B_{mask}$$ } | Maximum distance in 3D between the treatment mask structure and the external body contour of each patient|
| Chamber neck distance  | $$\bar{B}_{mask}$$ | Considers the average value of the distribution of the 3D distances|
| Haussdorff neck distance | $$\sigma_{B_{mask}}$$ | Corresponds to the standard deviation of the distribution of the distances|
| Maximum 2D neck distance| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Median 2D neck distance| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Average 2D neck distance| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Minimum 3D radius| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Maximum 3D radius| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Average 3D radius| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Ratio between minimum and maximum 3D radius | $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Average cross-sectional neck area| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Surface area| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|
| Compactness| $$V^{air}_{Body-to-mask}$$ | Amount of space that is between the external body contour of the HN region and the treatment mask contour|


#### Submandibular-related metrics
| Metric  | Symbol | Definition |
| ------------- | ------------|-----------|
| Area  | $$A_{sub}$$ | Total space covered by the submandibular body contour plane|
| Minimum 2D radius  | $$R_{min}$$ | Minimum 2D radius between the center of the CT sim submandibular body contour plane and a submandibular body contour plane |
| Maximum 2D radius| $$R_{max}$$ | Maximum 2D radius between the center of the CT sim submandibular body contour plane and a submandibular body contour plane|
| Average 2D radius| $$R_{avg}$$ | Average 2D radius that can be made between the center of the CT sim submandibular body contour plane and a submandibular body contour plane|
| Ratio between minimum and maximum 2D radius| $$\varphi^{2D}_{R}$$ | Calculated by the ratio between $$R_{max}$$ and $$R_{min}$$. This metric characterizes the symmetry of the submandibular plane contour|

## Dependencies
  *  Python>=3.6
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
1. By pulling the latest version from GitHub. Please note that your Python installation should be 3.6 or later. 
```
pip install git+https://github.com/kildealab/HN-GeoNatomyX.git
```

## Usage and Examples

## How to Cite 
If you publish any work using this package, please make sure you acknowledge us by citing the following paper: Rios-Ibacache O., Manalad J., O'Sullivan-Steben K., Poon E., at al. Quantification of head and neck cancer patients’ anatomical changes: prediction of radiotherapy replanning need.
## License 
This project is provided under the GNU GLPv3 license to preserve open-source access to any derivative works. See the LICENSE file for more information.
## References
* Barker, J. L., Jr, Garden, A. S., Ang, K. K., O'Daniel, J. C., Wang, H., Court, L. E., Morrison, W. H., Rosenthal, D. I., Chao, K. S., Tucker, S. L., Mohan, R., & Dong, L. (2004). Quantification of volumetric and   geometric changes occurring during fractionated radiotherapy for head-and-neck cancer using an integrated CT/linear accelerator system. International journal of radiation oncology, biology, physics, 59(4), 960–970. https://doi.org/10.1016/j.ijrobp.2003.12.024
* Figen, M., Çolpan Öksüz, D., Duman, E., Prestwich, R., Dyker, K., Cardale, K., Ramasamy, S., Murray, P., & Şen, M. (2020). Radiotherapy for Head and Neck Cancer: Evaluation of Triggered Adaptive Replanning in Routine Practice. Frontiers in oncology, 10, 579917. https://doi.org/10.3389/fonc.2020.579917
* van Beek, S., Jonker, M., Hamming-Vrieze, O., Al-Mamgani, A., Navran, A., Remeijer, P., & van de Kamer, J. B. (2019). Protocolised way to cope with anatomical changes in head & neck cancer during the course of radiotherapy. Technical innovations & patient support in radiation oncology, 12, 34–40. https://doi.org/10.1016/j.tipsro.2019.11.001
* Patrick, H. M., & Kildea, J. (2022). Technical note: rtdsm-An open-source software for radiotherapy dose-surface map generation and analysis. Medical physics, 49(11), 7327–7335. https://doi.org/10.1002/mp.15900


    

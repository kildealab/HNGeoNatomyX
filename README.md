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

Head and neck cancer patients undergoing radiotherapy may experience significant anatomical changes due to weight loss and tumor shrinkage. These changes can impact the effectiveness of the initial treatment plan, potentially necessitating treatment replanning. However, ad hoc replanning requires additional clinical staff time, which can lead to suboptimal and stressful treatment planning. Furthermore, currently, there is no established method for determining the total amount of anatomical variation in the head and neck region to decide whether replanning is necessary. This research aimed to identify and create metrics based on patient anatomical structures that can describe the anatomical alterations that patients may experience throughout the treatment and influence decisions regarding treatment replanning. These parameters were used to develop a machine learning classification model to predict if patients would likely undergo replanning. Based on the 3D shape and 2D contours of structures, we defined 43 parameters. We developed **HN-GeoNatomyX**, an extraction pipeline to automatically calculate the defined 43 parameters over the course of the radiotherapy treatment.

## Features
  ### Semi-automatic contouring tool for the treatment mask
  ![Optional Text](Mask_treatment.jpg)
  ### Code for extracting the 43 parameters stored in scripts

## Dependencies
  *  Python>=3.6
  *  [rtdsm](https://github.com/kildealab/rtdsm)==0.1.1
  *  [scipy](https://scipy.org/)>=1.6.0
  *  [skimage](https://scikit-image.org/)>=0.24.0
  *  [numpy](https://numpy.org/)>=1.19.5
  *  [pyvista](https://docs.pyvista.org/) >=0.31.3
  *  [alphashape](https://alphashape.readthedocs.io/en/latest/index.html) >=1.3.1
  *  [pydicom](https://pydicom.github.io/pydicom/stable/) >=3.0.1
  *  [sklearn](https://scikit-learn.org/stable/index.html)>=1.5.2
  *  [shapely](https://shapely.readthedocs.io/en/stable/)>=2.0.2
  *  [opencv-python](https://opencv.org/get-started/)>=4.10.0.84
  *  [json](https://docs.python.org/3/library/json.html)>=3.13.5

## Installation
```
pip install git+https://github.com/kildealab/HN-GeoNatomyX.git
```

1. Clone the repository
   ```
   git clone https://github.com/kildealab/HN-GeoNatomyX.git
   ```
2. Install dependencies
   ```
   cd HN-GeoNatomyX
   pip install -r requirements.txt
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


    

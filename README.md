# Quantification of head and neck cancer patients’ anatomy and longitudinal analysis: prediction of radiotherapy replanning need

Head and neck cancer patients undergoing radiotherapy may experience significant anatomical changes due to weight loss and tumor shrinkage. These changes can impact the effectiveness of the initial treatment plan, potentially necessitating treatment replanning. However, ad hoc replanning requires additional clinical staff time, which can lead to suboptimal and stressful treatment planning. Furthermore, currently, there is no established method for determining the total amount of anatomical variation in the head and neck region to decide whether replanning is necessary. 

This research aimed to identify and create metrics based on patient anatomical structures, that can describe the anatomical alterations that patients may experience throughout the treatment and influence decisions regarding treatment replanning. These parameters were used to develop a machine learning classification model to predict if patients would likely undergo replanning. This study included 150 head and neck cancer patients treated at the McGill University Health Centre. 

Based on the 3D shape and 2D contours of structures, we defined 43 parameters. We performed a univariate analysis using linear regression analysis and obtained the variation of each parameter concerning initial values, to provide significant insights for evaluating replanning. 

# This repository contains:
  1. Code for extracting the 43 parameters stored in scripts
  2. Semi-automatic contouring tool for the treatment mask

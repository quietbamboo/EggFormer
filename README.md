Nondestructive in ovo sexing of Hy-Line Sonia eggs by EggFormer using hyperspectral imaging
===

The code implements the "Nondestructive in ovo sexing identification of Hy-Line Sonia eggs by eggformer using hyperspectral imaging".
The structure of model is detailed as follows.
![image](./img/model.png)


Requirements
---
```` Python
# torch: 1.10.2   cuda: 11.3  
pip install -r requirements.txt
conda env create -f in_ovo_sexing.yml
````


How to use it?
---

### Before using it
This code uses preprocessing methods for the dataset:   
Firstly, it is necessary to extract the Region of Interest (ROI) from the hyper-spectral images.   
The parts outside the ROI are set to 0, so that the calculation of the average spectrum within the ROI will be more accurate.   
Secondly, each spectral image is saved in a file format similar to .pkl for easy reading.   
The files of one certain day should be named in the format of ID_gender and all dataset files are placed in one folder named with xd.  
The pretrained weights of ViT-Base with ImageNet-21k can be downloaded at https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth


### Select smooth methods with Random Forest(RF)
```` Python 
# the results are saved in ./rf_select_smooth_methods/results/
python ./rf_select_smooth_methods/run_rf_select.py --input_root_dir Your_root_dir  --days [xd]
````


### Select significant wavelengths by RF, PCA, SPA, CARS
```` Python
# RF
python ./find_feature_bands/run_rf.py  --input_root_dir Your_root_dir  --days [xd]

# PCA
python ./find_feature_bands/run_pca.py  --input_root_dir Your_root_dir  --days [xd]

# SPA
python ./find_feature_bands/run_spa.py  --input_root_dir Your_root_dir  --days [xd]

# CARS
# The flag save_mat_flag should be True first to save mat files and then use Matlab to generate results during CARS.
# and then modify save_mat_flag to False to calculate PLS-DA predicting results.
python ./find_feature_bands/run_cars.py --input_root_dir Your_root_dir  --days [xd]
````


### EggFormer
```` Python
# EggFormer
python ./eggformer/run_eggformer.py --mode eggformer  --freeze-layers True  --weights the path of pretrained weights of ViT-Base with ImageNet-21k  --input-days [number of days]

# ViT-Base
python ./eggformer/run_eggformer.py --mode vit  --freeze-layers False  --weights ''   --input-days [number of days]

# ViT-Base with pretrained weights
python ./eggformer/run_eggformer.py --mode vit  --freeze-layers True  --weights the path of pretrained weights of ViT-Base with ImageNet-21k   --input-days [number of days]
````


### Evaluate EggFormer on even-number days from 0 to 14
```` Python
python ./eggformer/run_eggformer.py --mode eggformer  --freeze-layers True  --weights the path of pretrained weights of ViT-Base with ImageNet-21k  --input-days [0, 2, 4, 6, 8, 10, 12, 14]
````


### Grad-CAM for heatmap
```` Python
python ./eggformer/show_GradCam.py  ----eggFormer-weights the path of weights of best model you trained
````

Contact Information:
---
Junxian Huang: [jim@njau.edu.cn](jim@njau.edu.cn)<br/>
Chengming Ji: [jicm@njau.edu.cn](jicm@njau.edu.cn)

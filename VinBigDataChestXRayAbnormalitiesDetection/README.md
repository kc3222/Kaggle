# Kaggle VinBigData Chest X-Ray Abnormalities Detection
## 10th-place Solution Source Code

**Discussion:** https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229629

**Final model weights:** https://www.kaggle.com/kc3222/vinbigdata-final-models

**Final model submissions:**
* ensemble/yolov5x_fold0_1_2_3_4_768_conf_0.01_d2_r101fpn3x_054999_vfnet_r101fpn_8020_fold0_1_2_3_4_wbf_skbthr_0.03_v45_submission.csv: 0.260 Public Leaderboard & 0.293 Private Leaderboard

**How to rerun:**
* Step 1: Download necessary datasets for training from public links (noted in input folder)
* Step 2: Change paths in each notebooks based on your environment
* Step 3: Start training
* Step 4: Infer each trained model
* Step 5: Ensemble inferences in ensemble folder

**Inference notebooks**
* yolov5: https://www.kaggle.com/kc3222/final-vinbigdata-cxr-ad-yolov5-v4-0-infer
* vfnet: https://www.kaggle.com/kc3222/mmdet-pytorch-framework-infer-vfnet
* fasterrcnn: https://www.kaggle.com/kc3222/final-vinbigdata-detectron2-prediction
* 2 class filter for yolov5 and vfnet (inference for fasterrcnn already has 2 class filter): https://www.kaggle.com/kc3222/final-vinbigdata-2-class-filter

**How to reproduce final submissions:**
* Step 1: Download outputs from yolov5, vfnet and fasterrcnn inference notebooks into /ensemble/model_inferences/
* Step 2: Run 2 class filter for each yolov5 and vfnet output, and download the results into /ensemble/model_inferences/
* You can skip step 1 and 2 by downloading the outputs of the 2-class filter notebook and the outputs of the fasterrcnn notebook.
* Step 3: Run merge_bboxes_v45.py (rename files if neccessary)

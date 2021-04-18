# %%
'''
Create Labels dataset for Yolov5
Example: https://www.kaggle.com/awsaf49/vinbigdata-yolo-labels-dataset
'''

# %%
import os
import numpy as np
import pandas as pd

# %%
wbf_merged_data = pd.read_csv('./wbf_train_data.csv')
train_meta_data = pd.read_csv('./train_meta.csv')

# %%
unique_imageids = wbf_merged_data['image_id'].unique().tolist()

# %%
def create_label_textfile(image_id, wbf_merged_data, label_dir):
    data = wbf_merged_data[wbf_merged_data['image_id'] == image_id]
    train_meta = train_meta_data[train_meta_data['image_id'] == image_id]
    width = train_meta['dim1'].values.tolist()[0]
    height = train_meta['dim0'].values.tolist()[0]
    class_id_lst = data['class_id'].values.tolist()
    x_mid_lst = ((data['x_max'].values + data['x_min'].values) / (width * 2)).tolist()
    y_mid_lst = ((data['y_max'].values + data['y_min'].values) / (height * 2)).tolist()
    width_lst = ((data['x_max'].values - data['x_min'].values) / width).tolist()
    height_lst = ((data['y_max'].values - data['y_min'].values) / height).tolist()
    f = open(os.path.join(label_dir, image_id+'.txt'), 'w')
    
    # Yolov5 Format (x_mid, y_mid, width, height)
    for class_id, x_mid, y_mid, width, height in zip(class_id_lst, x_mid_lst, y_mid_lst, width_lst, height_lst):
        line = str(class_id) + ' ' + str(x_mid) + ' ' + str(y_mid) + ' ' + str(width) + ' ' + str(height) + ' \n'
        f.write(line)
    f.close()

# %%
for i, imageid in enumerate(unique_imageids):
    label_dir = './labels'
    # if i == 1:
    #     break
    test = wbf_merged_data[wbf_merged_data['image_id'] == imageid]
    create_label_textfile(imageid, wbf_merged_data, label_dir)

# %%

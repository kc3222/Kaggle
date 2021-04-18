# %%
import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion

# %%
weights = [40, 40, 40, 40, 40, 35, 41.2, 41.5, 41.3, 40.5, 42.3, 40.3]
iou_thr = 0.5
iou_thr_large = 0.4
skip_box_thr = 0.03 # 0.0001 for non soft-nms
skip_box_thr_large = 0.03
sigma = 0.1

# %%
yolov5_fold0_tta_pred = pd.read_csv('./model_inferences/yolov5/yolov5x_fold0_finetune768_submission.csv')
yolov5_fold1_tta_pred = pd.read_csv('./model_inferences/yolov5/yolov5x_fold1_finetune768_submission.csv')
yolov5_fold2_tta_pred = pd.read_csv('./model_inferences/yolov5/yolov5x_fold2_finetune768_submission.csv')
yolov5_fold3_tta_pred = pd.read_csv('./model_inferences/yolov5/yolov5x_fold3_finetune768_2cls_filter_submission.csv')
yolov5_fold4_tta_pred = pd.read_csv('./model_inferences/yolov5/yolov5x_fold4_finetune768_2cls_filter_submission.csv')

detectron2_r101fpn3x_pred = pd.read_csv('./model_inferences/fasterrcnn/d2_r101fpn_2cls_filter_submission.csv')

vfnet_r101fpn_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_8020_v1_epoch18_2cls_filter_submission.csv')
vfnet_r101fpn_fold0_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_fold0_v3_epoch4_2cls_filter_submission.csv')
vfnet_r101fpn_fold1_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_fold1_v4_epoch18_2cls_filter_submission.csv')
vfnet_r101fpn_fold2_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_fold2_v4_epoch18_2cls_filter_submission.csv')
vfnet_r101fpn_fold3_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_fold3_v1_epoch25_2cls_filter_submission.csv')
vfnet_r101fpn_fold4_pred = pd.read_csv('./model_inferences/vfnet/vfnet_r101_fold4_v3_epoch18_2cls_filter_submission.csv')

test_meta = pd.read_csv('../input/test_meta.csv')
merged_df = pd.DataFrame(columns=['image_id', 'PredictionString'])
ensemble_filename = 'yolov5x_fold0_1_2_3_4_768_conf_0.01_d2_r101fpn3x_054999_vfnet_r101fpn_8020_fold0_1_2_3_4_wbf_skbthr_0.03_v45_submission.csv'

# %%
'''Weighted Boxes Fusion'''
image_id_lst = yolov5_fold4_tta_pred['image_id'].unique()
class_large = [0, 1, 3, 4, 12]

# %%
# Helper functions
def extract_data(data, img_height, img_width, class_large = [3, 5, 12]):
    boxes_large_lst = []
    scores_large_lst = []
    labels_large_lst = []
    boxes_normal_lst = []
    scores_normal_lst = []
    labels_normal_lst = []
    data_lst = data.split(' ')
    for i in range(0, len(data_lst), 6):
        label = int(data_lst[i])
        if label in class_large:
            labels_large_lst.append(int(data_lst[i]))
            scores_large_lst.append(float(data_lst[i + 1]))
            x_min = float(data_lst[i + 2]) / img_width
            y_min = float(data_lst[i + 3]) / img_height
            x_max = float(data_lst[i + 4]) / img_width
            y_max = float(data_lst[i + 5]) / img_height
            boxes_large_lst.append([x_min, y_min, x_max, y_max])
        else:
            labels_normal_lst.append(label)
            scores_normal_lst.append(float(data_lst[i + 1]))
            x_min = float(data_lst[i + 2]) / img_width
            y_min = float(data_lst[i + 3]) / img_height
            x_max = float(data_lst[i + 4]) / img_width
            y_max = float(data_lst[i + 5]) / img_height
            boxes_normal_lst.append([x_min, y_min, x_max, y_max])
    return boxes_large_lst, scores_large_lst, labels_large_lst, boxes_normal_lst, scores_normal_lst, labels_normal_lst

def convert_data_to_row(boxes, scores, labels):
    data_lst = []
    for i in range(len(boxes)):
        data_lst.append(str(int(labels[i])))
        data_lst.append(str(scores[i]))
        data_lst.append(str(boxes[i][0]))
        data_lst.append(str(boxes[i][1]))
        data_lst.append(str(boxes[i][2]))
        data_lst.append(str(boxes[i][3]))
    data = ' '.join(data_lst)
    return data

def get_height_width(image_id):
    # dim0: heigth, dim1: width
    height = test_meta[test_meta['image_id'] == image_id]['dim0'].values[0]
    width = test_meta[test_meta['image_id'] == image_id]['dim1'].values[0]
    return height, width

def scale_data(boxes, img_height, img_width):
    boxes[:, 0] = boxes[:, 0] * img_width
    boxes[:, 1] = boxes[:, 1] * img_height
    boxes[:, 2] = boxes[:, 2] * img_width
    boxes[:, 3] = boxes[:, 3] * img_height
    return boxes

# %%
def wbf(image_id):
    img_height, img_width = get_height_width(image_id)
    boxes_lst, scores_lst, labels_lst = [], [], []
    boxes_large_lst, scores_large_lst, labels_large_lst = [], [], []

    yolov5_fold0_tta_data = yolov5_fold0_tta_pred[yolov5_fold0_tta_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(yolov5_fold0_tta_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    yolov5_fold1_tta_data = yolov5_fold1_tta_pred[yolov5_fold1_tta_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(yolov5_fold1_tta_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    yolov5_fold2_tta_data = yolov5_fold2_tta_pred[yolov5_fold2_tta_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(yolov5_fold2_tta_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    yolov5_fold3_tta_data = yolov5_fold3_tta_pred[yolov5_fold3_tta_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(yolov5_fold3_tta_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    yolov5_fold4_tta_data = yolov5_fold4_tta_pred[yolov5_fold4_tta_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(yolov5_fold4_tta_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    detectron2_r101fpn3x_data = detectron2_r101fpn3x_pred[detectron2_r101fpn3x_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(detectron2_r101fpn3x_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_data = vfnet_r101fpn_pred[vfnet_r101fpn_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_fold0_data = vfnet_r101fpn_fold0_pred[vfnet_r101fpn_fold0_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_fold0_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_fold1_data = vfnet_r101fpn_fold1_pred[vfnet_r101fpn_fold1_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_fold1_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_fold2_data = vfnet_r101fpn_fold2_pred[vfnet_r101fpn_fold2_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_fold2_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_fold3_data = vfnet_r101fpn_fold3_pred[vfnet_r101fpn_fold3_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_fold3_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    vfnet_r101fpn_fold4_data = vfnet_r101fpn_fold4_pred[vfnet_r101fpn_fold4_pred['image_id'] == image_id]['PredictionString'].values[0]
    model_boxes_large_lst, model_scores_large_lst, model_labels_large_lst, model_boxes_lst, model_scores_lst, model_labels_lst = extract_data(vfnet_r101fpn_fold4_data, img_height, img_width)
    boxes_large_lst.append(model_boxes_large_lst)
    scores_large_lst.append(model_scores_large_lst)
    labels_large_lst.append(model_labels_large_lst)
    boxes_lst.append(model_boxes_lst)
    scores_lst.append(model_scores_lst)
    labels_lst.append(model_labels_lst)

    boxes_large, scores_large, labels_large = weighted_boxes_fusion(boxes_large_lst, scores_large_lst, labels_large_lst, weights=weights, iou_thr=iou_thr_large, skip_box_thr=skip_box_thr_large)
    boxes, scores, labels = weighted_boxes_fusion(boxes_lst, scores_lst, labels_lst, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes_large = scale_data(boxes_large, img_height, img_width)
    boxes = scale_data(boxes, img_height, img_width)

    # Join two boxes
    final_boxes = np.append(boxes, boxes_large, axis=0)
    final_scores = np.append(scores, scores_large, axis=0)
    final_labels = np.append(labels, labels_large, axis=0)

    merged_data = convert_data_to_row(final_boxes, final_scores, final_labels)
    merged_data = pd.DataFrame([[image_id, merged_data]], columns=['image_id', 'PredictionString'])
    return merged_data

# %%
# Test
test_wbf = wbf(image_id_lst[0])

# %%
for image_id in image_id_lst:
    merged_data = wbf(image_id)
    merged_df = merged_df.append(merged_data, ignore_index=True)

# %%
merged_df.to_csv(ensemble_filename, index=False)

# %%

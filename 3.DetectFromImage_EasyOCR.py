import warnings
warnings.filterwarnings("ignore")

print("Importing Necessary Libraries")

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

import easyocr

import csv
import uuid

def main():

    print("Imports Completed")

    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        }

    files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    #################### Things to setup ########################################
    MAX_BOXES_TO_DRAW = 5
    MIN_SCORE_THRES = 0.6
    IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars423.png')
    region_threshold = 0.05
    detection_threshold = 0.7
    LOAD_CHECKPOINT = 'ckpt-101'
    #################### /Things to setup ########################################
    
    
    ################### DEFINE FUNCTIONS #########################################
    
    ######################## Detections #########################
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES):
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'] + label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates = True,
                    max_boxes_to_draw = MAX_BOXES_TO_DRAW,
                    min_score_thresh = MIN_SCORE_THRES,
                    agnostic_mode = False)
        
        return image_np_with_detections, detections

   ############ Apply OCR to Detection ############
    # region_threshold = 0.05
    # detection_threshold = 0.7

    def filter_text(region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1]
        
        plate = [] 
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        return plate

    def ocr_it(image, detections, detection_threshold, region_threshold):
        
        # Scores, boxes and classes above threhold
        scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]
        classes = detections['detection_classes'][:len(scores)]
        
        # Full image dimensions
        width = image.shape[1]
        height = image.shape[0]
        
        # Apply ROI filtering and OCR
        for idx, box in enumerate(boxes):
            roi = box*[height, width, height, width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            reader = easyocr.Reader(['en'])
            ocr_result = reader.readtext(region)
            
            text = filter_text(region, ocr_result, region_threshold)
            
            plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            plt.show()
            print(text)
            return text, region, roi, boxes, scores, classes
        
    ########### Save Results  ########### 
    def save_results(text, region, roi, boxes, scores, classes, csv_filename, folder_path):
        img_name = '{}.jpg'.format(uuid.uuid1())
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        cv2.imwrite(os.path.join(folder_path, img_name), region)
        
        with open(csv_filename, mode = 'a', newline = '') as f:
            csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            csv_writer.writerow([img_name, text, roi, boxes, scores, classes])

     ################### /DEFINE FUNCTIONS #######################################
    

    ############ Load pipeline config and build a detection model ################
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config = configs['model'], is_training=False)

    ############ Restore checkpoint ##############################################
    ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], LOAD_CHECKPOINT)).expect_partial()


    ##############################################################################
    ################## Detections ################################################
    ##############################################################################
    # MAX_BOXES_TO_DRAW = 5
    # MIN_SCORE_THRES = 0.6
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    #IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars107.png')

    #rcParams['figure.figsize'] = 20, 10
    img = cv2.imread(IMAGE_PATH)

    image_np_with_detections, detections = detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES)

    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # plt.show()

    # for i in detections.keys():
    #     print(i)


    #############################################################################
    ################## OCR ######################################################
    #############################################################################        


    try:
        text, region, roi, boxes, scores, classes = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
        save_results(text, region, roi, boxes, scores, classes, './output/DetectionFromImages.csv', './output/Detection_From_Images')
    except:
        print("Not Found")
        pass

    print("Finished..............................................")

if __name__ == '__main__':
    main() 


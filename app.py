import warnings
warnings.filterwarnings("ignore")

print("Starting.............................................................................\n")
print("Importing Necessary Libraries\n")

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoTransformerBase
import tempfile
import av
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
import time
from zipfile import ZipFile
import base64

import easyocr

import csv
import uuid

print("Imports Completed\n")

#st.set_page_config(layout = "wide")
st.set_page_config(page_title = "Automatic Number Plate Recognition and EasyOCR", page_icon="🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#################### Title #####################################################
#st.title('Automatic Number Plate Recognition and EasyOCR')
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Automatic Number Plate Recognition and EasyOCR</h3>", unsafe_allow_html=True)
#st.markdown('---') # inserts underline
#st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
st.markdown('#') # inserts empty space

#################### SideBar ####################################################
activity = ["Detect From Image", "Detect From Live Feed", "Detect From Video File"]
choice = st.sidebar.selectbox('Chose An Activity', activity)
st.sidebar.subheader("Parameters")

#################### Parameters to setup ########################################
MAX_BOXES_TO_DRAW = st.sidebar.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 5)
MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.6)
region_threshold = st.sidebar.slider('Region Threshold', min_value = 0.0, max_value = 1.0, value = 0.05)
detection_threshold = st.sidebar.slider('OCR Detection Threshold', min_value = 0.0, max_value = 1.0, value = 0.7)
LOAD_CHECKPOINT = 'ckpt-101'
#################### /Parameters to setup ########################################


folderImages = os.path.join(".", "output", "Detection_From_Images")
folderRealTime = folderToZip1 = os.path.join(".", "output", "Detection_From_RealTimeFeed")
folderVideos = os.path.join(".", "output", "Detection_From_Videos")

detectFromImagesZipFile = os.path.join(".", "output", "Detection_From_Images.zip")
detectFromRealTimeFeedZipFile = os.path.join(".", "output", "Detection_From_RealTimeFeed.zip")
detectFromVideosZipFile = os.path.join(".", "output", "Detection_From_Videos.zip")


DEMO_VIDEO = "./samples/sampleVideo0.mp4"
# DEMO_VIDEO = "./samples/sampleVideo1.mp4"
# DEMO_VIDEO = "./samples/sampleVideo2.mp4"
DEMO_PIC = "./samples/Cars3.png"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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

print("Processing.............\n")



############ Load pipeline config and build a detection model ################
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config = configs['model'], is_training = False)

############ Restore checkpoint ##############################################
ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], LOAD_CHECKPOINT)).expect_partial()

############ Category Index ##################################################
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

################### DEFINE FUNCTIONS #######s##################################

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

############ Detections #########################

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

@st.cache()
def detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES):
    image_np = np.array(img)
    #print("Image NP: \n", image_np)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype = tf.float32)
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

@st.cache()
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

@st.cache(allow_output_mutation = True)
def ocr_it(image, detections, detection_threshold, region_threshold):
    
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
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
        
        # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        # plt.show()
        print(text)
        return text, region, roi, boxes, scores, classes
    
########### Save Results  #########################
@st.cache()
def save_results(text, region, roi, boxes, scores, classes, csvFilePath, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    cv2.imwrite(os.path.join(folder_path, img_name), region)
    
    with open(csvFilePath, mode = 'a', newline = '') as f:
        csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text, roi, boxes, scores, classes])

def main():   
    
    ##############################################################################
    ################## Streamlit #################################################
    ##############################################################################
    
    if choice == "Detect From Image":
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        #csv_filename_images = './output/Detection_From_Images/{}.csv'.format(uuid.uuid1())
        #csv_filename_images = os.path.join(folderImages, 'Detection_From_Images.csv')
        csv_filename_images = './output/Detection_From_Images/Detection_From_Images.csv'
        
        ###################### Image File Upload ######################################
        
        uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "png"])

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            #resized = cv2.resize(img,(224, 224))
            # Now do something with the image! For example, let's display it:
            st.sidebar.image(img, channels = "RGB")
        else:
            img = cv2.imread(DEMO_PIC)
            st.sidebar.text("Uploaded/Default Pic")
            st.sidebar.image(img, channels = "RGB")
            #st.info("Upload an Image")

        ###################### Detections And OCR #############################################
        
        if st.button('Detect'):
            st.markdown("## Output")
            with st.spinner(text = 'In progress..............'):
                try:
                    
                    #time.sleep(5)
                    image_np_with_detections, detections = detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES)
                    st.image(image_np_with_detections, channels = "RGB")
                    
                    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
                    # plt.show()

                    # for i in detections.keys():
                    #     print(i)
                except Exception as e:
                    st.write("Error in Detection")
                    st.error(e)
            with st.spinner(text = 'In progress..............'):
                try:
                    text, region, roi, boxes, scores, classes = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
                    save_results(text, region, roi, boxes, scores, classes, csv_filename_images, folderImages)
                    #st.write(text, region)
                    
                    
                    col1, col2, col3, col4, col5 = st.columns((1, 1, 2, 1, 1))
                    with col1:
                        st.write("Extracted Text: ", text)
                    with col2:
                        st.write("Region of Interest: ", roi)
                    with col3:
                        st.write("Bounding Boxes: ", boxes)
                    with col4:
                        st.write("Scores: ", scores)
                    # with col5:
                    #     st.write("Classes: ", classes)
                    
                    df = pd.read_csv(csv_filename_images)
                    st.dataframe(df)
                    st.success('Done')
                except Exception as e:
                    st.write("Error in OCR")
                    st.error(e)
                    pass
            
            #Zip and Downloads        
            st.sidebar.markdown('## Output')   
            try:
                with st.spinner("Please Wait....."):
                    # Zip and download
                    zipObj = ZipFile(detectFromImagesZipFile, 'w')
                    if zipObj is not None:
                        # Add multiple files to the zip
                        files = os.listdir(folderImages)
                        #print("folder Images:", folderImages)
                        for filename in files:
                            eachFile = os.path.join(folderImages, filename)
                            zipObj.write(eachFile)
                        zipObj.close()
            except Exception as e:
                st.write("Error in Zip and Download")
                st.error(e)
            
            try:
                with open(detectFromImagesZipFile, 'rb') as f:
                    st.sidebar.download_button('Download (.zip) Img2Text', f, file_name = 'Detection_From_Images.zip')
            except Exception as e:
                print(e)
                
                
    if choice == "Detect From Live Feed":
        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "User Other Camera"), index = 0)
        
        if selectedCam == "User Other Camera":
            selectedCam = int(1)
        else:
            selectedCam = int(0)
        
        #st.sidebar.write("Select Camera is ", selectedCam)
        
        stframe = st.empty()
        cap = cv2.VideoCapture(selectedCam)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Width: ", width, "\n")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Height: ", height, "\n")
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        print("FPS Input: ",fps_input, "\n")
        
        startStopCam = st.checkbox("Record/Stop")
         
        if startStopCam:
            st.info("Recording in Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            #filename = './output/RealTimeFeed/{}.mp4'.format(uuid.uuid1())
            realTimeVideofilename = os.path.join(folderRealTime, 'RealTimeFeed.mp4')
            #realTimeVideofilename = './output/Detection_From_RealTimeFeed/RealTimeFeed.mp4'
            csv_filename_RealTime = os.path.join(folderRealTime, 'Detection_From_RealTime.csv')
            #csv_filename_RealTime = './output/Detection_From_RealTimeFeed/Detection_From_RealTime.csv'
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (width, height)
            
            VideoOutPut = cv2.VideoWriter(realTimeVideofilename, codec, fps_input, resolution)
            
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    #Calculations for getting FPS manually
                    newTime = time.time()
                    fps = int(1/(newTime - prevTime))
                    prevTime = newTime
                    #print("FPS: ", fps)
                    #image_np = np.array(frame)
                    
                    stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    image_np_with_detections, detections = detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES)
                    #For Testing
                    #text, region, roi, boxes, scores, classes = 0, 0, 0, 0, 0, 0
                    
                    try: 
                        text, region, roi, boxes, scores, classes = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
                        save_results(text, region, roi, boxes, scores, classes, csv_filename_RealTime, folderRealTime)
                    except Exception as e:
                        print(e)
                    
                    time.sleep(1/fps_input)
                    print("Sleeping: ", 1/fps_input, "\n")
                    VideoOutPut.write(image_np_with_detections)
                    stframe.image(cv2.resize(image_np_with_detections, (width, height)), channels = 'BGR', use_column_width = True)
                else:
                    break
            
            vid.release()
            VideoOutPut.release()
        
        #Zip and Downloads    
        st.sidebar.markdown('## Output')  
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(detectFromRealTimeFeedZipFile, 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(folderRealTime)
                    for filename in files:
                        eachFile = os.path.join(folderRealTime, filename)
                        zipObj.write(eachFile)
                    zipObj.close()
                    
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(detectFromRealTimeFeedZipFile, 'rb') as f:
            st.sidebar.download_button('Download (.zip) Real Time Feed', f, file_name = 'RealTimeFeed.zip')
        
        #Print File Names
        try:
            st.sidebar.write(csv_filename_RealTime)
            st.sidebar.write(csv_filename_RealTime)
            print("Real Time .cvs File Name: ", csv_filename_RealTime, "\n")
            print("Real Time .mp4 File Name: ", realTimeVideofilename, "\n")
        except:
            pass
    
       
    if choice == "Detect From Video File":
        
        st.set_option('deprecation.showfileUploaderEncoding', False)
        stframe = st.empty()
        #video_file_buffer = st.sidebar.file_uploader("Upload a video", type = [ "mp4", "mov",'avi','asf', 'm4v' ])
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type = [ "mp4", "m4v"])
        tffile = tempfile.NamedTemporaryFile(delete = False)

        if not video_file_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
        else:
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)
            
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #st.write(width)
        print("Width: ", width, "\n")
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #st.write(height)
        print("Height: ", height, "\n")
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))
        #st.write(fps_input)
        print("FPS Input: ",fps_input, "\n")
        
        #st.sidebar.markdown("## Output")
        #st.sidebar.text("Default/Uploaded Video")
        st.sidebar.markdown("**Default/Uploaded Video**")
        st.sidebar.video(tffile.name)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Text**")
            kpi2_text = st.markdown("0")
            kpi2 = "Change Later"

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")
        
        with kpi4:
            st.markdown("**Image Height**")
            kpi4_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        
        detect = st.checkbox("Detect/Stop")
        
        if detect:
            #st.info("Detection in Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            #filename = './output/Detection_From_Videos/{}.mp4'.format(uuid.uuid1())
            videofilename = os.path.join(folderVideos, 'outVideoFile.mp4')
            #videofilename = './output/Detection_From_Videos/outVideoFile.mp4'
            csv_filename_Video = os.path.join(folderVideos, 'Detection_From_Videos.csv')
            #csv_filename_Video = './output/Detection_From_Videos/Detection_From_Videos.csv'
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (width, height)
           
            VideoOutPut = cv2.VideoWriter(videofilename, codec, fps_input, resolution)
            
            while vid.isOpened():
                ret, img = vid.read()
                if ret:
                    #Calculations for getting FPS manually
                    newTime = time.time()
                    fps = int(1/(newTime - prevTime))
                    prevTime = newTime
                    
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img_height, img_width, _ = img.shape
                    #print("Image Shape: \n", img.shape)
                    # print("Type: \n", type(img))
                    
                    image_np_with_detections, detections = detect_func(img, MAX_BOXES_TO_DRAW, MIN_SCORE_THRES)
                    
                    try: 
                        text, region, roi, boxes, scores, classes = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
                        save_results(text, region, roi, boxes, scores, classes, csv_filename_Video, folderVideos)
                    except Exception as e:
                        print(e)
                        # vid.release()
                        # VideoOutPut.release()
                        
                    time.sleep(1/fps_input)
                    print("Sleeping: ", 1/fps_input, "\n")
                    VideoOutPut.write(image_np_with_detections)        
                            
                    #Dashboard            
                    kpi1_text.write(f"<h3 style='text-align: left; color: red;'>{int(fps_input)}</h3>", unsafe_allow_html=True)
                    #kpi1_text.write(f"<h4 style='text-align: left; color: red;'>{int(fps)}</h4>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h4 style='text-align: left; color: red;'>{text}</h4>", unsafe_allow_html=True)
                    kpi3_text.write(f"<h4 style='text-align: left; color: red;'>{width}</h4>", unsafe_allow_html=True)
                    kpi4_text.write(f"<h4 style='text-align: left; color: red;'>{height}</h4>", unsafe_allow_html=True)
                    #Display on Dashboard
                    stframe.image(cv2.resize(image_np_with_detections, (width, height)), channels = 'BGR', use_column_width = True)
                    
                else:
                    break
        
            vid.release()
            VideoOutPut.release()
            
        #Zip and Downloads
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(detectFromVideosZipFile, 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(folderVideos)
                    for filename in files:
                        eachFile = os.path.join(folderVideos, filename)
                        zipObj.write(eachFile)
                    zipObj.close()
                    
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(detectFromVideosZipFile, 'rb') as f:
            st.download_button('Download (.zip) Video2Text', f, file_name = 'DetectionFromVideos.zip')
                
        #Print File Names
        try:
            st.write(csv_filename_Video)
            st.write(videofilename)
            print("Video .cvs File Name: ", csv_filename_Video, "\n")
            print("Video .mp4 File Name: ", videofilename, "\n")
        except:
            pass
            
    print("END ............................................................................")
          


if __name__ == "__main__":
    main()
# Automatic Number Plate Recognition and EasyOCR


## Step 1: Download Kaggle Dataset

 - Download Dataset via kaggle API or manually from [KAGGLE](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
 - Follow the instructions for Downloading via [Kaggle API](https://www.kaggle.com/docs/api)
 - Run `kaggle datasets download -d andrewmvd/car-plate-detection` in the terminal where the project folder is

## Step 2: Move Dataset into a Training and Testing Partition

- Run `python ./1.PrepareDataset.py` locally

## Step 3: Training and Detection

- Upload file `ANPR_and_EasyOCR_ColabRun_v1.ipynb` in Google Colaboratory
- Remember to Upload `archive.tar` file of prepared dataset in the step 2 when running `ANPR_and_EasyOCR_ColabRun_v1.ipynb` in the directory `Tensorflow/workspace/images`
 <table style="width:100%">
- Note down the latest checkpoint in `Tensorflow\workspace\models\CUSTOM_MODEL_NAME\` e.g. `\Tensorflow\workspace\models\my_ssd_mobnet`. This will be required to enter in scripts `3.DetectFromImage_EasyOCR.py`, `3.DetectFromImage_EasyOCR.py`, `3.DetectFromImage_EasyOCR`, `app.py`
- Download the compressed file of trained model

## Step 4: Run the Streamlit UI
 - Run `streamlit run app.py`
 
 Click on the image to see full view.
  <tr>
    <td><img src="https://i.imgur.com/9xqhmps.png" width="500px" height=150px/></td>
    <td><img src="https://i.imgur.com/wVvrsVv.png" width="500px" height=150px/></td>
    <td><img src="https://i.imgur.com/lPeUea7.png" width="500px" height=150px/></td>
   <td><img src="https://i.imgur.com/rhOMFNk.png" width="500px" height=150px/></td>
   </tr>
</table>
 

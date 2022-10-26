import os
from random import choice
import shutil
import zipfile
import tarfile

def main():
    print("Creating Paths........................")
    WORKING_DIR = os.getcwd()
    print(WORKING_DIR)
    DATASET_PATH = os.path.join(WORKING_DIR, 'dataset')
    print(DATASET_PATH)
    IMAGES_PATH = os.path.join(DATASET_PATH, 'images_with_annotations')
    print(IMAGES_PATH)

    print("Making Directories.....................")
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    
    print("Unziping Downloaded Dataset.............")
    with zipfile.ZipFile("car-plate-detection.zip") as zipref:
        zipref.extractall()
    
    print("Moving images and annotation files.......")
    # Move Files image files from './Images' to 'Dataset/Images'
    for files in os.listdir('images'):
        EX_PATH = os.path.join('images', files)
        NEW_PATH = os.path.join(IMAGES_PATH, files)
        os.replace(EX_PATH, NEW_PATH)

    # Move Files annotation files from './annotations' to 'Dataset/Images'
    for files in os.listdir('annotations'):
        EX_PATH = os.path.join('annotations', files)
        NEW_PATH = os.path.join(IMAGES_PATH, files)
        os.replace(EX_PATH, NEW_PATH)
    
    print("Deleting images and annotations dirs.......")    
    os.rmdir("./annotations")
    os.rmdir("./images")
    
    #arrays to store file names
    imgs =[]
    xmls =[]

    print("Spliting into Test and Train Dirs...........")
    #SETUP DIRECTORIES FOR TRAIN, TEST
    trainPath = os.path.join(IMAGES_PATH, 'train')
    if not os.path.exists(trainPath):
        os.makedirs(trainPath)
        

    testPath = os.path.join(IMAGES_PATH, 'test')
    if not os.path.exists(testPath):
        os.makedirs(testPath)
                
    crsPath = IMAGES_PATH #dir where images and annotations stored

    #setup ratio (val ratio = rest of the files in origin dir after splitting into train and test)
    train_ratio = 0.8
    test_ratio = 0.2 


    #total count of imgs
    totalImgCount = len(os.listdir(crsPath))/2
    #totalImgCount

    #soring files to corresponding arrays
    for (dirname, dirs, files) in os.walk(crsPath):
        for filename in files:
            if filename.endswith('.xml'):
                xmls.append(filename)
            else:
                imgs.append(filename)
                

    #counting range for cycles
    countForTrain = int(len(imgs)*train_ratio)
    countForTest = int(len(imgs)*test_ratio)


    #cycle for train dir
    for x in range(countForTrain):

        imagFile = choice(imgs) # get name of random image from origin dir
        fileXml = imagFile[:-4] +'.xml' # get name of corresponding annotation file

        #move both files into train dir
        shutil.move(os.path.join(crsPath, imagFile), os.path.join(trainPath, imagFile))
        shutil.move(os.path.join(crsPath, fileXml), os.path.join(trainPath, fileXml))

        #remove files from arrays
        imgs.remove(imagFile)
        xmls.remove(fileXml)


    #cycle for test dir   
    for x in range(countForTest):

        imagFile = choice(imgs) # get name of random image from origin dir
        fileXml = imagFile[:-4] +'.xml' # get name of corresponding annotation file

        #move both files into train dir
        shutil.move(os.path.join(crsPath, imagFile), os.path.join(testPath, imagFile))
        shutil.move(os.path.join(crsPath, fileXml), os.path.join(testPath, fileXml))

        #remove files from arrays
        imgs.remove(imagFile)
        xmls.remove(fileXml)

    # #move remaining files to test folder
    xmls = []
    imgs = []
    for files in os.listdir(crsPath):
        if files[-4:] == ".xml":
            xmls.append(files)
        if files[-4:] == ".png":
            imgs.append(files)
        
      
    if len(xmls) == len(imgs):
        for i in range(len(xmls)):
            EX_PATH = os.path.join(IMAGES_PATH, xmls[i])
            NEW_PATH = os.path.join(IMAGES_PATH, 'test', xmls[i])
            os.replace(EX_PATH, NEW_PATH)
            EX_PATH = os.path.join(IMAGES_PATH, imgs[i])
            NEW_PATH = os.path.join(IMAGES_PATH, 'test', imgs[i])
            os.replace(EX_PATH, NEW_PATH)

    #summary information after splitting
    print('Total images: ', totalImgCount)
    print('Images in train dir:', len(os.listdir(trainPath))/2)
    print('Images in test dir:', len(os.listdir(testPath))/2)
    
    #Compressing
    print("Compressing....................")
    ARCHIVE_PATH = os.path.join('dataset', 'archive.tar.gz')
    testPath = os.path.join('dataset', 'images_with_annotations', 'test')
    trainPath = os.path.join('dataset', 'images_with_annotations', 'train')
    
    with tarfile.open(ARCHIVE_PATH, "w:gz") as tarhandle:
        tarhandle.add(testPath)
        tarhandle.add(trainPath)
                
    
    print("Finished........................")
    
    
if __name__ == '__main__':
    main() 

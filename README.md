***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
 - **IMAGE_FILENAME**: The filename of each image.
 - **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 



## Requirements
Code is written with python 3 and to use the code, you need to first install the following python packages:

```Shell
pip install notebook
pip install numpy
pip install pandas
pip install opencv-python
pip install pytorch
pip install matplotlib
pip install seaborn
pip install tensorflow==2.0.0  
pip install tensorflow-gpu==2.0.0  
pip install keras==2.3.1  
pip install pandas==0.25.3  
pip install numpy==1.17.4  
pip install scikit-learn==0.22  
pip install scikit-image==0.16.2  
pip install matplotlib==3.1.2  
```

Preprocessing:

- Create custom generator with Balanced Batch as well as more augmentation functions than keras ImageDataGenerator supports 
  (e.g. gaussian noise). 
- The images a somewhat blurry, adding resolution improvement may help.

Model and model optimization:  

- Add background_id as another input to the network. 
  !Caution, this may decrease generalizability of the model if new background classes would be introduced.  
- Add learning rate decay.
- Add early stopping to prevent overfitting.
- Create a Deep CNN architecture from scratch, train it on more epochs.

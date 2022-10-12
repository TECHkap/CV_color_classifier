"""
@author: TECHkap
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.image as mpimg
import random

# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["zoom_in", "zoom_out"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

# This function should take in an RGB image and return a new, standardized version
# 560 height x 320 width image size (px x px)
def standardize_input(image):
    
    # Resize image and pre-process so that all "standard" images are the same size  
    standard_im = cv.resize(image, (560, 320))
    
    return standard_im


# Examples: 
# encode("zoom_out") should return: 1
# encode("zoom_in") should return: 0
def encode(label):
        
    numerical_val = 0
    if(label == 'zoom_out'):
        numerical_val = 1
    # else it is night and can stay 0
    
    return numerical_val

# using both functions above, standardize the input images and output labels
def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # Create a numerical label
        binary_label = encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, binary_label))
        
    return standard_list

def videotoframe(FILE, OUTPUT_PATH):
    """
    Input:
        FILE: Video File, from which the frames are extracted
        OUTPUT_PATH: path, where the frames are saved
    Output:
        No Return
        Folder with frames is created
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    cap = cv.VideoCapture(FILE)
    
    
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    
    idx = 0    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            idx += 1
            if idx % 7 != 0:
                    continue
            cv.imwrite(OUTPUT_PATH + '/' + str(idx) + '.png', frame)
        else:
            break
    print ("ALL FRAMES ARE SAVE CORRECTLY!")
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print (length)
    cap.release()
    return

#def greendetector(IMAGE_RGB):
#    image = cv.imread (IMAGE_RGB)
#    
#    # Make a copy of the image
#    image_copy = np.copy(image)
#    
#    #resize
#    image_copy = image_copy[0:300,0:500]
    
def RGB_separator(IMAGE_RGB):

    
    r = IMAGE_RGB[:,:,0]
    g = IMAGE_RGB[:,:,1]
    b = IMAGE_RGB[:,:,2]
    
    
    
    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('Standardized image')
    ax1.imshow(IMAGE_RGB)
    ax2.set_title('R channel')
    ax2.imshow(r, cmap='gray')
    ax3.set_title('G channel')
    ax3.imshow(g, cmap='gray')
    ax4.set_title('B channel')
    ax4.imshow(b, cmap='gray')    
    
    plt.show()
    
    
    return (IMAGE_RGB, r, g, b)

def HSV_separator(IMAGE_RGB):

    IMAGE_HSV = cv.cvtColor(IMAGE_RGB, cv.COLOR_RGB2HSV)
    
    h = IMAGE_HSV[:,:,0]
    s = IMAGE_HSV[:,:,1]
    v = IMAGE_HSV[:,:,2]
    
    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('Standardized image')
    ax1.imshow(IMAGE_RGB)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')    
    
    plt.show()
    
    
    return (IMAGE_RGB, h, s, v)
    
def GREEN_brightness (IMAGE_RGB):
    
    
    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(IMAGE_RGB[:,:,1])
    area = 320*560.0  # pixels
    
    # find the avg
    avg = sum_brightness/area
#    print ("AVERAGE", avg)
    
    return avg

def estimate_label(IMAGE_RGB, threshold):
    
    ## TODO: extract green average brightness feature from an RGB image 
    # Use the avg brightness feature to predict a label (0, 1)
    predicted_label = 0
       
    ## TODO: Return the predicted_label (0 or 1) based on whether the avg is 
    # above or below the threshold
    avg = GREEN_brightness(IMAGE_RGB)
    
    if avg > threshold :
        predicted_label = 1
    if avg <= threshold:
        predicted_label = 0
    
    return predicted_label 

def get_misclassified_images(test_images, threshold):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im, threshold)

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

if __name__ == "__main__":
    

    """
    1st step: Transform video to Frames for examples Train data corresponds 
    of video frames all 10 frames and test data corresponds of video 
    frames all 13 frame
    
    2nd Step: manually label the frames in 2 Folders for the classifier
    
    3rd Step: create data set for training
    4rd Step: standardize the training set
    5th Step: Train the classifier with Robustness study
    
    5th Step: create data set for testing
    6th Step: Standardize the testing set
    7th Step: check the classifier on test data
    """
#==================== TRANSFORM VIDEO TO FRAME ================================
    filename = "/share/csv-id-db/AIRBAG_METHODENENTW_ENDPOS/01_Doku/Schulung/Lessons_Test/Eintracht_FCB.mp4"
    path = '/share/csv-id-db/AIRBAG_METHODENENTW_ENDPOS/01_Doku/Schulung/Lessons_Test/Frames_all'    
#    videotoframe(filename, path)
    
    image_dir_training = '/share/csv-id-db/AIRBAG_METHODENENTW_ENDPOS/01_Doku/Schulung/Lessons_Test/image_dir_training'
    image_dir_test = '/share/csv-id-db/AIRBAG_METHODENENTW_ENDPOS/01_Doku/Schulung/Lessons_Test/image_dir_test'

    
#==================== CREATE TRAINING DATA SET ================================    
    # Load training data
    IMAGE_LIST = load_dataset(image_dir_training)
#    print(IMAGE_LIST)
    
    #zoom in bis index 207 und zoom in bis 405
    image_index = 10
    selected_image = IMAGE_LIST[image_index][0]
    selected_label = IMAGE_LIST[image_index][1]
    
       
#==================== SET STANDARDIZATION ================================    
    # Standardize all training images
    STANDARDIZED_LIST = standardize(IMAGE_LIST)
    total_train = len(STANDARDIZED_LIST)

    selected_image = STANDARDIZED_LIST[image_index][0]
    selected_label = STANDARDIZED_LIST[image_index][1]
    
#    # Display image and data about it
#    plt.imshow(selected_image)
#    print ("pred :",estimate_label(selected_image, 0.43) ,"real :", selected_label)
#    plt.show()


#==================== TRAIN CLASSIFIER ================================     
    ROBUSTNESS_LIST = []
    threshold_range = np.arange(0.4,0.5,0.01)
    for rob in threshold_range:
        random.shuffle(STANDARDIZED_LIST)
        MISCLASSIFIED = get_misclassified_images(STANDARDIZED_LIST, rob)
        num_correct = total_train - len(MISCLASSIFIED)
        accuracy = num_correct/total_train
        ROBUSTNESS_LIST.append((rob,accuracy))
    
    ROBUSTNESS_LIST = np.array(ROBUSTNESS_LIST)
    
    plt.plot(ROBUSTNESS_LIST[:,0], ROBUSTNESS_LIST[:,1], 'bo--', markersize = 4)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.show()


#==================== CREATE TEST DATA SET ================================    
    # Load test data
    TEST_IMAGE_LIST = load_dataset(image_dir_test)
    
#==================== SET STANDARDIZATION ================================     
    # Standardize the test data
    STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
    total_test = len(STANDARDIZED_TEST_LIST)
    

#====================  CLASSIFIER PREDICTION ================================  
    best_threshold = ROBUSTNESS_LIST[:,0][np.argmax(ROBUSTNESS_LIST[:,1])]
    print ("BEST THRESHOLD: ", best_threshold)
    # Shuffle the standardized test data
    random.shuffle(STANDARDIZED_TEST_LIST)          
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST, best_threshold)
    # Accuracy calculations
    num_correct = total_test - len(MISCLASSIFIED)
    accuracy = num_correct/total_test
    
    print('Accuracy: ' + str(accuracy))
    print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total_test))
    
#====================  VISUALIZE FALSE PREDICTION  ============================     
    num = 0
    test_mis_im = MISCLASSIFIED[num][0]
    for mis_im in range(10):
        print ("pred:", MISCLASSIFIED[mis_im][2], "real:", MISCLASSIFIED[mis_im][1])
        plt.imshow(MISCLASSIFIED[mis_im][0])
        plt.show()
        
    
#    im_standard, red, green, blue = RGB_separator(im_copy)
#    
##    im_standard, hue, saturation, value = HSV_separator(im_copy)
#    
    


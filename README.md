# ![Am I responsive](assets/images/amiresponsive.png)

### Deployed web aplication [Mildew Detection](https://pp5mildewdeiain-8e8f491a1401.herokuapp.com/)

## Table of Contents
1. [](#)
2. [](#)
3. [](#)
4. [](#)
5. [](#)
6. [](#)

## Business Requirements

### Business Requirement 1: 

The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.

### Business Requirement 2: 

The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

### Business Requirement 3:

We agreed with the client a degree of 97% accuracy.

### Business Requirement 4:

The client is interested in obtaining a prediction report of the examined leaves.

## Dashboard Design (Streamlit App User Interface)

### [Streamlit](https://streamlit.io/) was used to create the dashboard for easy uses and prenst data

#### Page One: Project Summary

The project summary page is the ladning page, it is the first page you will see. 
It in detail explains the General information, Project dataset and Business requirments.

#### Page Two: Cherryleaves Visualizer

On this page you can see the differences between average healthy and powdery-mildew leaves aswell as a image montage of healthy or powdery-mildew leaves.

#### Page Three: Powdery mildew detection

On this page you can dowload a photo from the link provided or if you have a picture of a cherry leaf you can drag and drop it onto the page, it will the give you a pridition of the leaf with a option to dowload a report of the prediction.

#### Page Three: Hypothesis

Here you will find a hypothesis of how to identify a healthy cherry leaf from a powdery-mildew leaf

#### Page Three: Ml Performance Metrics

This page has the perfomance of the model been used for the dashboard going into depth about the model and the performance of it.

## The goal

The goal is to meet all the buisness requirements aswell as display it in a format that is easaly readable to anyone using the dasboard but aswell as have a section for users who are more intrested of the indepth results of the model

## Dataset Content (for model v11)

1. The data was gatherd from [kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) and dowloaded into zipfolders that cosited of healthy leaf photos aswell as powdery-mildew photos.
I thenproceeded to unzip the folders first clean the data to insure I only have images and then deleted fifty percent of the data from both files and this was the rusult of the image count 

   - Folder: healthy - has 2104 image files
   - Folder: healthy - has 0 non-image files
   - Folder: powdery_mildew - has 2104 image files
   - Folder: powdery_mildew - has 0 non-image files
   - Folder: healthy - Deleted 1052 images based on 50% deletion.
   - Folder: powdery_mildew - Deleted 1052 images based on 50% deletion.
   - Folder: healthy - has 1052 images remaining
   - Folder: powdery_mildew - has 1052 images remaining

2. I split the data into train, valaidation and test sets as follows

   - The training set is divided into a 0.70 ratio of data.
   - The validation set is divided into a 0.10 ratio of data.
   - The test set is divided into a 0.20 ratio of data.

3. I then resized all the images 

   - Mean width of images: 256 
   - Mean height of images: 256

4. I then got the mean and variability of images per label

   - <details>
     <summary>mean and variability:</summary>

     ![Average var healthy](outputs/v11/avg_var_healthy.png)
     ![Average Powdery mildew](outputs/v11/avg_var_powdery_mildew.png)
     ![Average difference](outputs/v11/avg_var_healthy.png)

     </details>

5. I do a count of how many images there are in the tarin, test and validation stes.
   The count is as follow

   - Train - healthy: 736 images
   - Train - powdery_mildew: 736 images
   - Validation - healthy: 105 images
   - Validation - powdery_mildew: 105 images
   - Test - healthy: 211 images
   - Test - powdery_mildew: 211 images
   - <details>
     <summary>Bar graph displaying the amount of images in each set:</summary>

     ![Test, Train and validtion sets](outputs/v11/labels_distribution.png)

     </details>

6. I then agumented training and validation images in hopes the model will pick up on more paterns of the leaves.
   An example of this would look like this:

   - <details>
     <summary>Bar graph displaying the amount of images in each set:</summary>

     ![Agumented](assets/images/agumented.png)

     </details>

## Model (V11)

The model shows rapid improvement for both loss and accuracy within the first few epochs, the model then achives and maintains high accuracy and low loss indecating effective learning and good generalization whithout 
overfitting

<details>
<summary>Here is a Graph pf the model</summary>

![Model](outputs/v11/model_merged_acc.png)

</details>

**Loss**

  - The model shows a reapid reduction in traing and validation losses int he first few epochs indecating that the model is learning efectively
  - The model losses then remains close to zero after the first few epochs showing that the model in not overfitting
  - The model losses stabelizes but there are some sings of fluctuctuation wich indicats some overfitting but not significant enough for me to deem this model not acceptable 
  - Overall the model demostrates excellent performance with low value losses 
  - <details>
    <summary>Here is a Graph pf the models training losses</summary>

    ![Training losses](outputs/v11/model_training_losses.png)

    </details>

**Accuracy**

  - The model shows rapid increase in accuracy in both validation and traing within the firs few epochs
  - There is no significant overfitting as the modeld stabilazes quickly with high accuaracy
  - The close alingment of both trainig and validation sugest good performance of the model
  - Overall the model shows excelent accuracy and high perfomance as well as mainaining this troughout the process
  - <details>
    <summary>Here is a Graph pf the models training accuracy</summary>

    ![Training accuracy](outputs/v11/model_training_acc.png)

    </details>

**Confusion Matrix**

Here we can see the model is highly accurate, correctly predicting a vast majority of instances in both classes. There are very few errors (Only two false negatives and no false positives) 

 - <details>
    <summary>Here is the Cofusion Matrix reults</summary>

    ![Cofusion Matrix](outputs/v11/confusion_matrix.png)

    </details>

**Optimizer** 

I used SGD as the optomizer fot this model as I have found better results using it, this the code I usesd.
For the loss I went wit binary_crossentopy with metrics as ccuaracy

  - <details>
    <summary>Code Snippit of optimizer</summary>

    ![Code Snippit](assets/images/optimizer.png)

    </details>

**Model Code explenation**

- For this model(v11) I have used five convolutional starting of with 16 filters increasing is size I found that this would stabelize the model.
- This model has a dropout set to 0.5 wich prevents too much overfittig or underfitting of this moddel
- A batch size of one was used
- Patience was set to 3 to insure the model would stop at the right time to prevent overfitting
- This model dense was set at 128 I found better performans at this value

 - <details>
    <summary>Code Snippits of the model</summary>

    ![Model Snippet](assets/images/modelsnipit.png)
    ![Batch size Snippit](assets/images/batchsize.png)
    ![Patience Snippit](assets/images/patience.png)

    </details>

## Trial and error

When creating this model it did not come without complicatition (Overfittin then underfitting) this is the elevnth model trhou trail and error is these only way I have acheved thes results I have used three difrent optomizers adam, RMS porp and SGD.
I have also used difrent patciance values, densety values aswel as difrent convolutional lairs.

At first I ran a few models but found I did not realy understand how the model was performing in regards to the relation to accuracy and value loss aswell as what classes it should improve on to solve this I have added more graphs to display the results in a better more readable way for me to see what the model is actualy dong and how I could improve on it.

Here is how each model performed :

  - <details>
    <summary>v1</summary>

    ![Model acc](outputs/v1/model_training_acc.png)
    ![Model loss](outputs/v1/model_training_losses.png)

    </details>
  - <details>
    <summary>v2</summary>

    ![Model acc](outputs/v2/model_training_acc.png)
    ![Model loss](outputs/v2/model_training_losses.png)

    </details>
  - <details>
    <summary>v3</summary>

    ![Model acc](outputs/v3/model_training_acc.png)
    ![Model loss](outputs/v3/model_training_losses.png)

    </details>
  - <details>
    <summary>v4</summary>

    ![Model acc](outputs/v4/model_training_acc.png)
    ![Model loss](outputs/v4/model_training_losses.png)
    ![Merged Graph](outputs/v4/model_merged_acc.png)

    </details>
  - <details>
    <summary>v5</summary>

    ![Model acc](outputs/v5/model_training_acc.png)
    ![Model loss](outputs/v5/model_training_losses.png)
    ![Merged Graph](outputs/v5/model_merged_acc.png)

    </details>
  - <details>
    <summary>v6</summary>

    ![Model acc](outputs/v6/model_training_acc.png)
    ![Model loss](outputs/v6/model_training_losses.png)
    ![Merged Graph](outputs/v6/model_merged_acc.png)
    ![Confusion Matrix](outputs/v6/confusion_matrix.png)

    </details>
  - <details>
    <summary>v7</summary>

    ![Model acc](outputs/v7/model_training_acc.png)
    ![Model loss](outputs/v7/model_training_losses.png)
    ![Merged Graph](outputs/v7/model_merged_acc.png)
    ![Confusion Matrix](outputs/v7/confusion_matrix.png)

    </details>
  - <details>
    <summary>v8</summary>

    ![Model acc](outputs/v8/model_training_acc.png)
    ![Model loss](outputs/v8/model_training_losses.png)
    ![Merged Graph](outputs/v8/model_merged_acc.png)
    ![Confusion Matrix](outputs/v8/confusion_matrix.png)

    </details>
  - <details>
    <summary>v9</summary>

    ![Model acc](outputs/v9/model_training_acc.png)
    ![Model loss](outputs/v9/model_training_losses.png)
    ![Merged Graph](outputs/v9/model_merged_acc.png)
    ![Confusion Matrix](outputs/v9/confusion_matrix.png)

    </details>
  - <details>
    <summary>v10</summary>

    ![Model acc](outputs/v10/model_training_acc.png)
    ![Model loss](outputs/v10/model_training_losses.png)
    ![Merged Graph](outputs/v10/model_merged_acc.png)
    ![Confusion Matrix](outputs/v10/confusion_matrix.png)

    </details>
  - <details>
    <summary>v11</summary>

    ![Model acc](outputs/v11/model_training_acc.png)
    ![Model loss](outputs/v11/model_training_losses.png)
    ![Merged Graph](outputs/v11/model_merged_acc.png)
    ![Confusion Matrix](outputs/v11/confusion_matrix.png)

    </details>
    


## Hypothesis and validation

The model is well tuned and effectively balanced

It simultaneously shows a decrease and low final values for both validation and training losses, alongside with close alignment of the curves

The model has successfully learned the patterns in the training set and can generalize these patterns to unseen validation data without overfitting

## Bugs

1. On model v5 there is a bug where the confusion matrix does not display corectly as this was the first implimantastion of it I did and fixed it for futre models
 - <details>
    <summary>Here you can see it is displaying incorectly (Model v5)</summary>
    
    ![Confusion Matrix](outputs/v5/confusion_matrix.png)

    </details>

2. I would not describe this as a bug but as an isuue that can be attended to in the futre, I found evrytime I made a new model I could have created a global variable where I can change the version on one place rather than have to change it indevidualy at multiple difrent places   

## Futre development

For futre development there are a few idias that could be cosidred:

1. Create a dropdwon to not ony identify chery leaves but aswell as other crops for the farmer
2. Create database when multiple users start loging on and using the webb aplication it will grow the database and in return train the model more efectivley
3. Have a section that shows on a map where farmers are experancing crop decieses with a heatmap showing the intensety of the spead of the decieses. 

## Deployment
   
### Forking the Repository

### Making a local clone

## Technologies used

### Platforms

### Languages

### Main Data Analysis and Machine Learning Libraries

## Credits

### Content

### Media

### Code

### Acknowledgements



import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v3'

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    
    st.info(f"The model I used was version three, it held the best accuracy and had the least losses\n\n"
            f"The amount of images used and ratios are as follows\n\n"
            f"* train - healthy: 736 images\n"
            f"* train - powdery_mildew: 736 images\n"
            f"* validation - healthy: 105 images\n"
            f"* validation - powdery_mildew: 105 images\n"
            f"* test - healthy: 211 images\n"
            f"* test - powdery_mildew: 211 images\n\n"
            )
    
    st.info( f"I used image shape from v2, \n"
            f"a batch size of one, \n"
            f"with experimenting I found I got the best results with four Convolutional blocks \n"
            f"The filters were set as follows: \n\n"
            f"* First Convolutional Block: 16 \n"
            f"* Second Convolutional Block: 32 \n"
            f"* Third Convolutional Block: 64 \n"
            f"* Fourth Convolutional Block: 128 \n\n"
            f"Dropout was set to 0.5, \n"
            f"the flatten block density was set to 128, \n"
            f"and the optimizer that was used is 'adam' with a patience of 4 and a total of 25 epochs"
            )
    
    st.write(
       f"For more information, please visit and **read** the "
       f"[Project README file](https://github.com/IainJackson90/pp5-mildew-detection#readme).")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
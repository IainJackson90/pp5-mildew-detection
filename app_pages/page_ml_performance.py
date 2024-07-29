import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v11'


    st.write("### Train, Validation and Test Set: Labels Frequencies")
   
    st.info(f"The model I used was version eleven, it held the best accuracy and had the least losses\n\n"
            )
   
    st.info( f"I used image shape from v11, \n"
            f"a batch size of one, \n"
            f"with experimenting I found I got the best results with four Convolutional blocks \n"
            f"The filters were set as follows: \n\n"
            f"* First Convolutional Block: 16 \n"
            f"* Second Convolutional Block: 32 \n"
            f"* Third Convolutional Block: 64 \n"
            f"* Fourth Convolutional Block: 128 \n"
            f"* Fifth Convolutional Block: 256 \n\n"
            f"Dropout was set to 0.5, \n"
            f"the flatten block density was set to 128, \n"
            f"and the optimizer that was used is 'SGD' with a patience of 3 and a total of 25 epochs"
            )
   
    st.write(
       f"For more information, please visit and **read** the "
       f"[Project README file](https://github.com/IainJackson90/pp5-mildew-detection#readme).")
   
    st.write("---")
    st.info( f"The images are split up into train, validation and test sets as show in the bar graph\n\n"
             f"The amount of images used and ratios are as follows\n\n"
             f"* train - healthy: 736 images\n"
             f"* train - powdery_mildew: 736 images\n"
             f"* validation - healthy: 105 images\n"
             f"* validation - powdery_mildew: 105 images\n"
             f"* test - healthy: 211 images\n"
             f"* test - powdery_mildew: 211 images\n\n"
            )
    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
   
    st.write("---")
    st.write("### Model History")
    st.info( f"In the model history we can see that the model did fifteen epoches out of twenty five before it was stopped by the patience of three\n\n"
            f" I have found stopping the model too late or too early would lead to the model overfitting or underfitting\n"
            )
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
        st.info( f"Here we can see that the model is able to learn and perform well on training data\n"
            f"and generalizes well to unseen data (Val_accuracy) which is crucial for real-world application \n"
            )
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
        st.info( f"\n"
            f"The graph shows that the model is learning effectively on the training data and also\n"
            f"that the validation loss is decreasing demonstrating the model's ability to generalize to unseen data\n"
            )
       
    st.write("---")
    st.write("### Accuracy and Loss model")
    mrg_model = plt.imread(f"outputs/{version}/model_merged_acc.png")
    st.image(mrg_model, caption='Accuracy and Loss model')
    st.write('### Model hypothesis')
    st.info( f"The graph indicates that the model is well tuned and effectively balanced\n\n"
            f"* It simultaneously shows a decrease and low final values for both validation and training losses,\n"
            f" alongside with close alignment of the curves\n\n"
            f"* The model has successfully learned the patterns in the training set and can generalize these patterns\n"
            f" to unseen validation data without overfitting \n"
            )
   
    st.write("---")
    st.write("### Confusion Matrix")
    cm_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(cm_matrix, caption='Confusion Matrix')
    st.info( f" Here we can see the model is highly accurate, correctly predicting a vast majority of\n"
            f" instances in both classes. There are very few errors (Only two false negatives and no false positives).\n"
            )
   
    st.write("---")
    st.write("### Generalized Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    st.info( f" Here we can see that we have met the business requirements of having a accuracy level of\n"
            f" ninety seven percent or above\n"
            )


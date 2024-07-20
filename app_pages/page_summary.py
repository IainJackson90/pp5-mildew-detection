import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a fungal disease in cherry trees caused by Podosphaera clandestina. "
        f"It forms a mildew layer of spores on the leaves, especially on new growth, slowing down the plant's growth and infecting the fruit, which leads to crop loss.\n\n"
        f"Infected and healthy leaves were examined. Signs of infection include:\n\n"
        f"* Light-green, circular lesions on the leaves\n"
        f"* White, cotton-like growth in the infected areas and on the fruits, reducing yield and quality."
        f" \n\n")

    st.warning(
        f"**Project Dataset**\n\n"
        f"The dataset uses can be found on [kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)\n\n"
        f"The available dataset contains healthy leaves and affected leaves "
        f"You can manually adjust the dataset size by specifying the percentage of data you want to use(Backend and is curently set at 50%) "
        f"The data is split into : \n\n" 
        f"* The training set is divided into a 0.70 ratio of data.\n"
        f"* The validation set is divided into a 0.10 ratio of data\n"
        f"* The test set is divided into a 0.20 ratio of data.\n"
        f" \n"
        f"Individually photographed against a neutral background."
        f"")

    st.success(
        f"The project has three business requirements:\n\n"
        f"1 - A study to visually differentiate a healthy from an infected leaf.\n\n"
        f"2 - An accurate prediction whether a given leaf is infected by powdery mildew or not. \n\n"
        f"3 - Download a prediction report of the examined leaves."
        )

    st.write(
        f"For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/IainJackson90/pp5-mildew-detection#readme).")
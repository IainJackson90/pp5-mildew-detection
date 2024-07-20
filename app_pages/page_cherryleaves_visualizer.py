import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_cherryleaves_visualizer_body():
    st.write("### Leaves Visualizer")
    st.info(
        f"A study that visually distinguishes between a cherry leaf affected by powdery mildew and a healthy one.")

    st.info(
        f"* A healthy cherry leaf is vibrant green, smooth, oval-shaped with finely serrated edges, free from spots or lesions, with clear, well-defined veins, and is flexible and consistent in size.\n\n"
        f"* We suspect that cherry leaves affected by powdery mildew exhibit distinct symptoms.\n"
        f"Initially, a light-green, circular lesion appears on either leaf surface, followed by a subtle white cotton-like growth in the infected area.\n\n"
        f"This characteristic must be translated into machine learning terms.\n"
        f"Images need to be 'prepared' before being fed to the model for optimal feature extraction and training.\n\n"
        f"When working with an image dataset, it's crucial to normalize the images before training a neural network.\n\n"
        f"To normalize an image, you'll need the mean and standard deviation of the entire dataset, which are calculated using a mathematical formula that considers the properties of an image."
        f"\n\n"
    )
    
    st.write(
       f"For more information, please visit and **read** the "
       f"[Project README file](https://github.com/IainJackson90/pp5-mildew-detection#readme).")
    
    version = 'v3'
    if st.checkbox("Difference between average and variability image"):
      
      avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
      avg_uninfected = plt.imread(f"outputs/{version}/avg_var_healthy.png")

      st.warning(
        f"We noticed that the average and variability images did not show "
        f"clear patterns to intuitively differentiate one from another. "
        f"However, leaves affected by mildew tend to show more white streaks in the center.")

      st.image(avg_powdery_mildew, caption='infected leaf - Average and Variability')
      st.image(avg_uninfected, caption='healthy leaf - Average and Variability')
      st.write("---")

    if st.checkbox("Differences between average infected and average healthy leaves"):
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          st.warning(
            f"We noticed this study didn't reveal patterns that allow us to intuitively differentiate one from another.")
          st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
      st.write("To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/mildew_dataset/cherry-leaves'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")
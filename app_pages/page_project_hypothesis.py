import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis")

    st.success(
        f"* We suspect cherry leaves exhibiting a white, powdery coating on the surface, "
        f"along with potential yellowing or curling, are likely to be affected by powdery mildew. \n\n"
        f"* An Image Montage shows that typically infected leaf has yellowing or curling features with a powdery coating on the surface . "
        f"Average Image, Variability Image and Difference between Averages studies did not reveal "
        f"any clear pattern to differentiate one from another."

    )
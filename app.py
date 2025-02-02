import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_cherryleaves_visualizer import page_cherryleaves_visualizer_body
from app_pages.page_powdery_mildew_detection import page_powdery_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Powdery - Mildew Detection for chery leaves")  # Create an instance of the app

# # Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Cherryleaves Visualizer", page_cherryleaves_visualizer_body)
app.add_page("Powdery mildew detection", page_powdery_mildew_detector_body)
app.add_page("Hypothesis", page_project_hypothesis_body)
app.add_page("Ml Performance Metrics",page_ml_performance_metrics)

app.run()  # Run the app
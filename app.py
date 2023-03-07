"""
Module used as a fe on hugging face
"""
import streamlit as st

x = st.slider('Select a value in this: ')
st.write(x, 'squared is', x * x)

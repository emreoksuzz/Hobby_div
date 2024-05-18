import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# Load the CSV data
df = pd.read_csv("/Users/emreoksuz/Downloads/Hobby_Data.csv")

# Set page configuration
st.set_page_config(
    page_title="Children Hobby Classifier",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8JaJKgR5hYXOcAKb8SkxExJAAYJAzij5uTrdggfflNQ&s",
)

# Title
st.title("Children Hobby Finder")

# Markdown description
st.markdown("Our company aims to predict children's hobbies based on their activities and various features, such as participation in academic projects, sports medals, and interest in arts.")

# Image
st.image("https://www.playwhereyoustay.org/wp-content/uploads/2021/10/pwys_s1.jpg")

# Further description
st.markdown("After recent advancements in the field of artificial intelligence, our company anticipates our assistance in creating a machine learning model tailored to their requirements.")
st.markdown("Furthermore, we seek a solution that allows predicting hobbies based on provided information. Let's address children's needs and deliver a powerful predictive tool.")
st.markdown("*Let's find their predisposition!*")

# Another image
st.image("https://www.wineanddesign.com/studio-shell/wp-content/uploads/sites/6/2023/05/peanut-into-picasso-new.jpg")

# Sidebar inputs
st.sidebar.markdown("**Choose** the features below to see the result!")
Name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
Surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
Olympiad_Participation = st.sidebar.selectbox("Academic Olympiad Participation",["Yes","No"])
Scholarship = st.sidebar.selectbox("Scholarship",["Yes","No"])
School = st.sidebar.selectbox("Loves School",["Yes","No"])
Fav_sub= st.sidebar.selectbox("Favourite Subject",["Math","Science","Language","History/Geography"])
Projects = st.sidebar.selectbox("Academic Projects",["Yes","No"])
Time_sprt = st.sidebar.number_input("Sport Time Spend in hours daily", min_value=0,max_value=6 ,format="%d")
Medals = st.sidebar.selectbox("Medals in Sport",["Yes","No"])
Act_sprt = st.sidebar.selectbox("Active in Sport",["Yes","No"])
Fant_arts = st.sidebar.selectbox("Loves Painting",["Yes","No"])
Won_arts = st.sidebar.selectbox("Won Art Competition",["Yes","No"])
Time_art= st.sidebar.number_input("Art Time Spend in hours daily", min_value=1,max_value=6 ,format="%d")

model_path = "/Users/emreoksuz/Downloads/XGBoost_model.joblib2"

# Load the trained model
model = load(model_path)

input_data = {
    'Olympiad_Participation_No': [1 if not Olympiad_Participation else 0],
    'Olympiad_Participation_Yes': [1 if Olympiad_Participation else 0],
    'Scholarship_No': [1 if Scholarship == "No" else 0],
    'Scholarship_Yes': [1 if Scholarship == "Yes" else 0],
    'School_No': [1 if not School else 0],
    'School_Yes': [1 if School else 0],
    'Fav_sub_Any language': [1 if Fav_sub == "Any language" else 0],
    'Fav_sub_History/Geography': [1 if Fav_sub == "History/Geography" else 0],
    'Fav_sub_Mathematics': [1 if Fav_sub == "Mathematics" else 0],
    'Projects_No': [1 if not Projects else 0],
    'Projects_Yes': [1 if Projects else 0],
    'Medals_No': [1 if not Medals else 0],
    'Medals_Yes': [1 if Medals else 0],
    'Act_sprt_No': [1 if not Act_sprt else 0],
    'Act_sprt_Yes': [1 if Act_sprt else 0],
    'Fant_arts_No': [1 if not Fant_arts else 0],
    'Fant_arts_Yes': [1 if Fant_arts else 0],
    'Won_arts_No': [1 if not Won_arts else 0],
    'Won_arts_Yes': [1 if Won_arts else 0],
    'Time_sprt': [Time_sprt],
    'Time_art': [Time_art]
}

# Convert input data to dataframe
input_df = pd.DataFrame(input_data)

pred = model.predict(input_df.values)

if st.sidebar.button("Submit"):
    st.info("You can find the result below.")

    results_df = pd.DataFrame({
        'Name': [Name],
        'Surname': [Surname],
        'Prediction': pred.tolist()

    })

    results_df["Prediction"] = results_df["Prediction"].replace({0: "Academics", 1: "Sports", 2: "Arts"})

    st.table(results_df)

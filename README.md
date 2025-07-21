# 🍽️ Restaurant Recommendation System

## Overview
This project builds a recommendation engine based on restaurant data, enabling personalized suggestions tailored to user preferences such as **city**, **cuisine**, **rating**, and **cost**. It includes a complete data pipeline for cleaning, encoding, and modeling, along with an interactive Streamlit application for user-friendly recommendations.

---

## Project Workflow

### 1. Data Cleaning and Preprocessing
- **Load** raw CSV restaurant data.
- **Remove duplicates** and handle missing values by imputation or removal.
- **Convert** numeric columns to proper formats.
- **Save** the cleaned dataset as `cleaned_data.csv`.

### 2. Feature Encoding
- **One-Hot Encode** categorical features (city, cuisine) using `OneHotEncoder`.
- **Exclude** high-cardinality fields like name for performance.
- **Combine** encoded features with numeric features.
- **Save** encoded features as `encoded_data.csv`.
- **Serialize** the encoder as `encoder.pkl` for use in production and app.

### 3. Recommendation Engine
- **Filter** restaurants based on user-selected city and cuisines.
- **Compute cosine similarity** between user preferences and encoded restaurant features.
- **Recommend** top-N restaurants mapped back to cleaned dataset for interpretability.

### 4. Streamlit Application
- **User interface** for input of city, cuisines, minimum rating, and cost.
- **Processes inputs**, queries encoded data, and displays restaurant recommendations.
- **Interactive**, real-time filtering and display of results.

---

## Technologies

- **Python 3.11+**
- **Data Processing:** `pandas`, `numpy`, `scikit-learn`
- **Recommendation Engine:** `scikit-learn` (cosine similarity)
- **Web App:** `streamlit`
- **Serialization:** `pickle` for encoder persistence

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/restaurant-recommendation.git
cd restaurant-recommendation

### 2. Install Dependencies

```bash
pip install -r requirements.txt
3. Prepare Input Data
Place your raw restaurant dataset CSV file (e.g., swiggy.csv) in the project directory.

Running the Project
🧹 Data Cleaning and Encoding
bash

python preprocess.py
Processes raw CSV to output:
cleaned_data.csv
encoded_data.csv
encoder.pkl
📊 Launch Interactive Recommendation App
bash

streamlit run app.py
Provides an interface for inputting preferences and viewing personalized restaurant recommendations.
Application Features
City and Cuisine Filtering

Recommendations are limited to user-selected city and cuisines for relevance.
Rating and Cost Constraints

Users can specify minimum rating and maximum cost to tailor suggestions.
Cosine Similarity Engine

Calculates similarity between user preference profile and restaurants in the dataset for personalized matches.
Interactive Streamlit Dashboard

Easy selection inputs with real-time display of recommended restaurants and links.
Directory Structure
plaintext

.
├── swiggy.csv                # Raw restaurant data
├── cleaned_data.csv          # Preprocessed cleaned data
├── encoded_data.csv          # One-hot encoded features with numeric columns
├── encoder.pkl               # Pickle file for trained OneHotEncoder
├── preprocess.py             # Data cleaning and encoding script
├── app.py                   # Streamlit recommendation application
├── requirements.txt          # Python package dependencies
└── README.md                 # This documentation file


Future Improvements
Implement clustering or collaborative filtering approaches for improved recommendations
Add more granular filters like location radius or dietary preferences
Support dynamic updates with streaming or API data sources
Deploy the Streamlit app on cloud platforms for easier access
Include user feedback loop to refine recommendations over time


Contact
For suggestions or collaboration, please contact: dhinakaransindhu96@gmail.com
📧 
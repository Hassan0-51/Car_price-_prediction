# Car Price Prediction Project

## Overview
This project aims to predict the selling price of used cars based on various features such as brand, mileage, fuel type, and condition. Using machine learning techniques, the project performs exploratory data analysis (EDA), preprocesses the dataset, trains a prediction model, and provides a Streamlit-based web application for interactive usage.

## Features
1. **Exploratory Data Analysis (EDA):**
   - Summary statistics
   - Distribution plots
   - Scatter plots for relationships between features
   - Bar plots for categorical data
   - Trends and grouped aggregations
2. **Data Preprocessing:**
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features
3. **Machine Learning Model:**
   - Linear Regression for prediction
   - Model evaluation using metrics like Mean Squared Error (MSE) and R-squared score
4. **Interactive Web Application:**
   - Built using Streamlit
   - Real-time predictions for car prices based on user inputs

## Dataset
The dataset contains information about various car features, including:
- **Brand**
- **Year**
- **Engine Size**
- **Fuel Type**
- **Transmission**
- **Mileage**
- **Condition**
- **Price**

### Data Preprocessing Steps
1. Added an `age` column calculated as `2024 - Year`.
2. Dropped irrelevant columns like `Year` and `Car ID`.
3. Separated numerical and categorical columns for targeted analysis and preprocessing.
4. Handled categorical data using Label Encoding.
5. Scaled numerical features using MinMaxScaler.

## Visualizations
- **Bar plots** to show the distribution of categorical variables.
- **Histograms** to analyze numerical features.
- **Scatter plots** for understanding relationships between features like mileage and price.
- **Box plots** for visualizing price distribution across categories.

## Model
- **Linear Regression**:
  - Trained on features like brand, fuel type, transmission, condition, and mileage.
  - Evaluated with:
    - Mean Squared Error (MSE)
    - R-squared (R²) score

## Streamlit Application
### Sections:
1. **Introduction**
   - Overview of the project and dataset.
2. **EDA**
   - Interactive visualizations and summaries of the dataset.
3. **Model**
   - Model training and evaluation results.
4. **Prediction**
   - Real-time price predictions based on user inputs.

## How to Run
1. Clone the repository.
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory.
   ```bash
   cd car-price-prediction
   ```
3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app.
   ```bash
   streamlit run app.py
   ```

## Results
- The model provides reasonably accurate predictions for car prices.
- Interactive visualizations enhance the understanding of the dataset.

## Future Work
- Experiment with advanced models like Random Forest or Gradient Boosting.
- Incorporate additional features like car color, location, or owner history.
- Optimize the application for better user experience.

## Acknowledgments
- Dataset sourced from Kaggle.
- Libraries used: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly, and Streamlit.


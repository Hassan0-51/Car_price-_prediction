import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('car_price_prediction_.csv')
    return df

# Cache the scaler
@st.cache_data
def fit_scaler(data, column):
    scaler = MinMaxScaler()
    scaler.fit(data[[column]])
    return scaler

# Load data
df = load_data()

# Sidebar Navigation
st.sidebar.title("Car Price Prediction App")
section = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Model", "Prediction"])

# Encoders and Scalers
encoder_brand = LabelEncoder()
encoder_fuel = LabelEncoder()
encoder_transmission = LabelEncoder()
encoder_condition = LabelEncoder()

# Fit encoders to the dataset
df['Brand'] = encoder_brand.fit_transform(df['Brand'])
df['Fuel Type'] = encoder_fuel.fit_transform(df['Fuel Type'])
df['Transmission'] = encoder_transmission.fit_transform(df['Transmission'])
df['Condition'] = encoder_condition.fit_transform(df['Condition'])

# Decode Functions
def decode_column(encoded_value, encoder):
    return encoder.inverse_transform([encoded_value])[0]

def decode_dataframe(df, columns, encoders):
    for col, encoder in zip(columns, encoders):
        df[col] = df[col].apply(lambda x: decode_column(x, encoder))
    return df

# Section: Introduction
if section == "Introduction":
    st.title("🚗 Car Price Prediction Project")
    st.write("""
    Welcome to the **Car Price Prediction App**! 🎉  
    This project utilizes **Machine Learning** to predict the selling price of used cars based on features like brand, mileage, fuel type, and condition.
    
    ### Features:
    - **Interactive Exploratory Data Analysis (EDA)**
    - **Machine Learning Model Training**
    - **Real-time Car Price Prediction**

    Get started by navigating through the sections on the left sidebar. 🚀
    """)

    st.write("### Dataset Overview")
    st.write(df.head())

# Section: EDA
if section == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")

    # Decode categorical columns for display
    decoded_df = decode_dataframe(df.copy(), ['Brand', 'Fuel Type', 'Condition'], 
                                  [encoder_brand, encoder_fuel, encoder_condition])

    # Summary statistics
    st.write("### Summary Statistics")
    st.write(decoded_df.describe())

    # Missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Selling Price Distribution
    st.write("### Selling Price Distribution")
    fig = px.histogram(df, x="Price", nbins=30, title="Selling Price Distribution")
    st.plotly_chart(fig)

    # Selling Price vs. Mileage
    st.write("### Selling Price vs. Mileage")
    fig = px.scatter(decoded_df, x="Mileage", y="Price", title="Selling Price vs. Mileage")
    st.plotly_chart(fig)

    # Selling Price by Brand
    st.write("### Selling Price by Brand")
    fig = px.box(decoded_df, x="Brand", y="Price", title="Selling Price by Brand")
    st.plotly_chart(fig)

    # Average, Cheapest, and Most Expensive Prices by Brand
    avg_price_by_brand = decoded_df.groupby('Brand')['Price'].mean()
    cheapest_brand = decoded_df.loc[decoded_df['Price'].idxmin()]
    expensive_brand = decoded_df.loc[decoded_df['Price'].idxmax()]

    st.write("#### Average Price by Brand")
    st.write(avg_price_by_brand)

    st.write("#### Cheapest Car")
    st.write(cheapest_brand)

    st.write("#### Most Expensive Car")
    st.write(expensive_brand)

    # Selling Price by Fuel Type
    st.write("### Selling Price by Fuel Type")
    avg_price_by_fuel = decoded_df.groupby('Fuel Type')['Price'].mean()
    st.write(avg_price_by_fuel)

    # Selling Price by Condition
    st.write("### Selling Price by Condition")
    avg_price_by_condition = decoded_df.groupby('Condition')['Price'].mean()
    st.write(avg_price_by_condition)

    # Prices Over the Years
    if 'Year' in decoded_df.columns:
        st.write("### Car Prices Over the Years")
        yearly_prices = decoded_df.groupby('Year')['Price'].mean().sort_index()
        fig = px.line(x=yearly_prices.index, y=yearly_prices.values, 
                      labels={'x': 'Year', 'y': 'Average Price'}, 
                      title="Trend of Car Prices Over the Years")
        st.plotly_chart(fig)

# Section: Model
if section == "Model":
    st.title("📈 Prediction Model")

    # Handle missing values
    df = df.dropna()

    # Fit scaler
    mileage_scaler = fit_scaler(df, "Mileage")
    df['Mileage'] = mileage_scaler.transform(df[['Mileage']])

    # Define features and target
    X = df.drop(columns=['Price', 'Car ID', 'Model'])
    y = df['Price']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.write("### Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared Score (R2): {r2:.2f}")

# Section: Prediction
if section == "Prediction":
    st.title("🔮 Car Price Prediction")

# User input for car features
    brand = st.selectbox("Select Brand:", options=encoder_brand.classes_)
    fuel_type = st.selectbox("Select Fuel Type:", options=encoder_fuel.classes_)
    transmission_type = st.selectbox("Select Transmission Type:", options=encoder_transmission.classes_)
    condition = st.selectbox("Select Condition:", options=encoder_condition.classes_)
    mileage = st.number_input("Enter Mileage (e.g., 15000):", min_value=0, max_value=1000000, value=50000, step=1000)

    # Preprocess user input
    brand_encoded = encoder_brand.transform([brand])[0]
    fuel_encoded = encoder_fuel.transform([fuel_type])[0]
    transmission_encoded = encoder_transmission.transform([transmission_type])[0]
    condition_encoded = encoder_condition.transform([condition])[0]

    # Normalize mileage input using the cached scaler
    mileage_scaler = fit_scaler(df, "Mileage")  # Ensure scaler is fitted and cached
    scaled_mileage = mileage_scaler.transform([[mileage]])[0][0]

    # Create feature vector with default values for missing features
    input_data = [[brand_encoded, fuel_encoded, transmission_encoded, condition_encoded, scaled_mileage, 0, 0]]  # Replace 0s with defaults if necessary

    # Prepare training data
    df = df.dropna()

    # Fit scaler
    mileage_scaler = fit_scaler(df, "Mileage")
    df['Mileage'] = mileage_scaler.transform(df[['Mileage']])

    # Define features and target
    X = df.drop(columns=['Price', 'Car ID', 'Model'])
    y = df['Price']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the price
    predicted_price = model.predict(input_data)[0]

    # Display predicted price
    st.write(f"### Predicted Selling Price: ₹{predicted_price:,.2f}")

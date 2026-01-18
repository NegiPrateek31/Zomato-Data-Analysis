# ZOMATO DATA ANALYSIS PROJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA

file_path = "zomatodata.csv"   
df = pd.read_csv(file_path, encoding='latin1')

# 2. BASIC CLEANING

df.columns = df.columns.str.strip()
df['Cuisines'] = df['Cuisines'].fillna("Not Specified")

# Map country codes
country_dict = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada",
    94: "Indonesia", 148: "New Zealand", 162: "Philippines",
    166: "Qatar", 184: "Singapore", 189: "South Africa",
    191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States"
}
df['Country'] = df['CountryCode'].map(country_dict)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create copies
df_raw = df.copy()       # for EDA
df_encoded = df.copy()   # for ML / encoding

# 3. EXPLORATORY DATA ANALYSIS

def eda_menu():
    while True:
        print("""
--- EDA MENU ---
1. Number of restaurants per country
2. Online delivery availability by country
3. Top 10 cuisines
4. Ratings distribution
5. Price range vs aggregate rating
6. Skewness & Kurtosis
7. Back
""")

        choice = input("Enter choice: ")

        if choice == '1':
            data = df_raw['Country'].value_counts()
            plt.figure(figsize=(12,6))
            sns.barplot(x=data.index, y=data.values)
            plt.xticks(rotation=45)
            plt.title("Restaurants per Country")
            plt.show()

        elif choice == '2':
            data = df_raw[df_raw['HasOnlinedelivery'] == 'Yes']['Country'].value_counts()
            plt.figure(figsize=(12,6))
            sns.barplot(x=data.index, y=data.values)
            plt.xticks(rotation=45)
            plt.title("Online Delivery by Country")
            plt.show()

        elif choice == '3':
            top_cuisines = df_raw['Cuisines'].value_counts().head(10)
            plt.figure(figsize=(10,6))
            sns.barplot(y=top_cuisines.index, x=top_cuisines.values)
            plt.title("Top 10 Cuisines")
            plt.show()

        elif choice == '4':
            plt.figure(figsize=(8,5))
            sns.histplot(df_raw['Aggregaterating'], bins=20, kde=True)
            plt.title("Rating Distribution")
            plt.show()

        elif choice == '5':
            plt.figure(figsize=(8,5))
            sns.boxplot(x='Pricerange', y='Aggregaterating', data=df_raw)
            plt.title("Price Range vs Rating")
            plt.show()

        elif choice == '6':
            numeric_cols = ['Votes', 'AverageCostfortwo', 'Aggregaterating']
            print("\nSkewness:\n", df_raw[numeric_cols].skew())
            print("\nKurtosis:\n", df_raw[numeric_cols].kurt())

        elif choice == '7':
            break

        else:
            print("Invalid choice")


# 4. ENCODING FOR MACHINE LEARNING

def encode_data():
    le = LabelEncoder()

    # Encode binary columns only
    binary_cols = ['HasOnlinedelivery', 'HasTablebooking']
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Extract main cuisine
    df_encoded['MainCuisine'] = df_encoded['Cuisines'].str.split(',').str[0]

    # One-hot encode selected columns
    df_encoded_final = pd.get_dummies(
        df_encoded,
        columns=['Country', 'MainCuisine'],
        drop_first=True
    )

    # Drop unnecessary text columns
    drop_cols = ['Cuisines', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose']
    for col in drop_cols:
        if col in df_encoded_final.columns:
            df_encoded_final.drop(col, axis=1, inplace=True)

    print("\nEncoding completed successfully!")
    print(df_encoded_final.head())

    return df_encoded_final

# 5. MAIN MENU

while True:
    print("""
--- MAIN MENU ---
1. Show first 5 rows
2. Show column names
3. Check missing values
4. Perform EDA
5. Encode data (for ML)
6. Exit
""")

    main_choice = input("Enter choice: ")

    if main_choice == '1':
        print(df_raw.head())

    elif main_choice == '2':
        print(df_raw.columns)

    elif main_choice == '3':
        print(df_raw.isnull().sum())

    elif main_choice == '4':
        eda_menu()

    elif main_choice == '5':
        df_final = encode_data()

    elif main_choice == '6':
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice")

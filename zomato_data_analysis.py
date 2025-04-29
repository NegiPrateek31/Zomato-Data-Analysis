import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = ""
df = pd.read_csv(file_path, encoding='latin1')

# Clean column names
df.columns = df.columns.str.strip()

# Fill missing Cuisines
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

while True:
    print("""
--- Zomato Data Main Menu ---
1. Display first 5 rows
2. Show column names
3. Check for null values
4. Drop duplicates
5. Encode categorical data
6. Exploratory Data Analysis (EDA)
7. Exit
""")

    choice = input("Enter your choice (1-7): ")

    if choice == '1':
        print(df.head())

    elif choice == '2':
        print(df.columns)

    elif choice == '3':
        print(df.isnull().sum())

    elif choice == '4':
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        print(f"{before - after} duplicate rows removed.")

    elif choice == '5':
        while True:
            print("""
--- Encode Categorical Data ---
1. Label Encoding
2. One-Hot Encoding
3. Back to Main Menu
""")
            sub_choice = input("Enter your choice (1-3): ")

            if sub_choice == '1':
                le = LabelEncoder()
                cat_cols = df.select_dtypes(include='object').columns
                for col in cat_cols:
                    df[col] = le.fit_transform(df[col])
                print("Label Encoding applied to all categorical columns.")
                print("\nEncoded Data (first 5 rows):")
                print(df.head())

            elif sub_choice == '2':
                df = pd.get_dummies(df, drop_first=True)
                print("One-Hot Encoding applied to categorical columns.")
                print("\nEncoded Data (first 5 rows):")
                print(df.head())

            elif sub_choice == '3':
                break

            else:
                print("Invalid encoding choice. Try again.")

    elif choice == '6':
        while True:
            print("""
--- EDA Submenu ---
1. Number of restaurants per country
2. Online delivery availability by country
3. Top 10 cuisines
4. Ratings distribution
5. Price range vs aggregate rating
6. Show skewness and kurtosis
7. Back to Main Menu
""")
            eda_choice = input("Enter EDA choice (1-7): ")

            if eda_choice == '1':
                country_restaurants = df['Country'].value_counts()
                plt.figure(figsize=(12, 6))
                sns.barplot(x=country_restaurants.index, y=country_restaurants.values)
                plt.title('Number of Restaurants per Country')
                plt.xticks(rotation=45)
                plt.ylabel('Restaurants')
                plt.xlabel('Country')
                plt.show()

            elif eda_choice == '2':
                online_delivery = df[df['HasOnlinedelivery'] == 'Yes']['Country'].value_counts()
                if not online_delivery.empty:
                    plt.figure(figsize=(10, 5))
                    sns.barplot(x=online_delivery.index, y=online_delivery.values)
                    plt.title('Online Delivery by Country')
                    plt.ylabel('Number of Restaurants with Online Delivery')
                    plt.xlabel('Country')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No restaurants offer online delivery.")


            elif eda_choice == '3':
                top_cuisines = df['Cuisines'].value_counts().head(10)
                sns.barplot(y=top_cuisines.index, x=top_cuisines.values)
                plt.title('Top 10 Cuisines')
                plt.ylabel('Cuisines')
                plt.xlabel('Count')
                plt.show()

            elif eda_choice == '4':
                sns.histplot(df['Aggregaterating'], bins=20, kde=True)
                plt.title('Rating Distribution')
                plt.xlabel('Aggregate Rating')
                plt.ylabel('Frequency')
                plt.show()

            elif eda_choice == '5':
                sns.boxplot(x='Pricerange', y='Aggregaterating', data=df)
                plt.title('Price Range vs Aggregate Rating')
                plt.xlabel("Price Range")
                plt.ylabel("Aggregate Rating")
                plt.show()

            elif eda_choice == '6':
                numeric_cols = df.select_dtypes(include=np.number).columns
                print("\n--- Skewness ---")
                print(df[numeric_cols].skew())
                print("\n--- Kurtosis ---")
                print(df[numeric_cols].kurt())

            elif eda_choice == '7':
                break

            else:
                print("Invalid EDA choice. Try again.")

    elif choice == '7':
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice. Try again.")

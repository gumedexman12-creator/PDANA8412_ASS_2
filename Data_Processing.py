#Data Processing Script 
import pandas as pd
import numpy as np

#Load the dataset
df = pd.read_csv("data/Books_Data_Clean.csv")

#Look inside dataset 
print("DATASET SHAPE")
print(df.shape)

print("\nFIRST FIVE ROWS")
print(df.head())

#Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

#Check data types
print("\nDATA TYPES")
print(df.dtypes)

#Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

#Doestarget variable exist?
if "bestseller" in df.columns:
    print("\nBESTSELLER DISTRIBUTION")
    print(df["bestseller"].value_counts(dropna=False))

    print("\nBESTSELLER PROPORTIONS")
    print(df["bestseller"].value_counts(normalize=True).round(3))
else:
    print("\nNo the 'bestseller' column does not exist yet,big boss!")

#Create the target variable

#75th percentile of units sold
threshold = df["units sold"].quantile(0.75)

print("\nBESTSELLER THRESHOLD")
print(threshold)

#Binary target variable
df["bestseller"] = (df["units sold"] > threshold).astype(int)

#Number of books in each class
print("\nBESTSELLER DISTRIBUTION")
print(df["bestseller"].value_counts().sort_index())

#Display the proportion of each class
print("\nBESTSELLER PROPORTIONS")
print(df["bestseller"].value_counts(normalize=True).sort_index().round(3))

#Display examples of the new target variable
print("\nSAMPLE OF TARGET VARIABLE")
print(df[["Book Name", "units sold", "bestseller"]].head(10))

#Investigating missing vals 
print("\nROWS WITH MISSING PUBLISHING YEAR")
print(
    df[df["Publishing Year"].isnull()]
    [["Book Name", "Author", "Publishing Year"]]
)

print("\nROWS WITH MISSING BOOK NAME")
print(
    df[df["Book Name"].isnull()]
    [["Book Name", "Author", "Publishing Year", "genre"]]
)

print("\nROWS WITH MISSING LANGUAGE CODE")
print(
    df[df["language_code"].isnull()]
    [["Book Name", "Author", "language_code", "bestseller"]]
    .head(10)
)
#Inestigate unusual publish years 
print("\nEARLIEST PUBLISHING YEARS")
print(
    df.nsmallest(15, "Publishing Year")
    [["Book Name", "Author", "Publishing Year"]]
)

print("\nLATEST PUBLISHING YEARS")
print(
    df.nlargest(10, "Publishing Year")
    [["Book Name", "Author", "Publishing Year"]]
)

#Start cleaning 

# Step 10: Clean column names
df.columns = df.columns.str.strip()

# Step 11: Handle invalid publishing years
print("\nNEGATIVE PUBLISHING YEARS BEFORE CLEANING")
print((df["Publishing Year"] < 0).sum())

# Replace negative years with missing values
df.loc[df["Publishing Year"] < 0, "Publishing Year"] = np.nan

print("\nMISSING PUBLISHING YEARS AFTER CLEANING")
print(df["Publishing Year"].isnull().sum())

# Step 12: Handle missing language codes
df["language_code"] = df["language_code"].fillna("Unknown")

print("\nLANGUAGE CODE VALUES AFTER CLEANING")
print(df["language_code"].value_counts())

# Step 13: Select initial model features
# Exclude identifiers, text fields and variables that
# could leak information about units sold.

features = [
    "Publishing Year",
    "language_code",
    "Author_Rating",
    "Book_average_rating",
    "Book_ratings_count",
    "genre",
    "sale price"
]

target = "bestseller"

# Create a separate modelling dataset
model_df = df[features + [target]].copy()

# Step 14: Check the prepared dataset
print("\nMODELLING DATASET SHAPE")
print(model_df.shape)

print("\nMODELLING DATASET COLUMNS")
print(model_df.columns.tolist())

print("\nMISSING VALUES IN MODELLING DATASET")
print(model_df.isnull().sum())

print("\nTARGET DISTRIBUTION AFTER CLEANING")
print(model_df[target].value_counts().sort_index())

#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Separate predictors (X) from the target (y)
X = model_df.drop(columns=["bestseller"])
y = model_df["bestseller"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING SET SHAPE")
print(X_train.shape)

print("\nTESTING SET SHAPE")
print(X_test.shape)

print("\nTRAINING TARGET DISTRIBUTION")
print(y_train.value_counts().sort_index())

print("\nTESTING TARGET DISTRIBUTION")
print(y_test.value_counts().sort_index())


# ==========================================
# STEP 16: IMPUTE MISSING VALUES
# ==========================================

# Calculate the median from the training data only
publishing_year_median = X_train["Publishing Year"].median()

# Fill missing publishing years in both sets
X_train["Publishing Year"] = (
    X_train["Publishing Year"].fillna(publishing_year_median)
)

X_test["Publishing Year"] = (
    X_test["Publishing Year"].fillna(publishing_year_median)
)

print("\nTRAINING MEDIAN PUBLISHING YEAR")
print(publishing_year_median)

print("\nMISSING VALUES AFTER IMPUTATION")
print("Training:", X_train.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())


# ==========================================
# STEP 17: ENCODE CATEGORICAL VARIABLES
# ==========================================

categorical_columns = [
    "language_code",
    "Author_Rating",
    "genre"
]

# Learn the categories from training data and apply them
# consistently to the test data
X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    dtype=int
)

# Make sure training and testing have the same columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("\nFEATURE COLUMNS AFTER ENCODING")
print(X_train.columns.tolist())

print("\nTRAINING FEATURES SHAPE AFTER ENCODING")
print(X_train.shape)

print("\nTESTING FEATURES SHAPE AFTER ENCODING")
print(X_test.shape)


# ==========================================
# STEP 18: STANDARDISE NUMERICAL FEATURES
# ==========================================

numerical_columns = [
    "Publishing Year",
    "Book_average_rating",
    "Book_ratings_count",
    "sale price"
]

# Calculate mean and standard deviation from training data
train_means = X_train[numerical_columns].mean()
train_stds = X_train[numerical_columns].std()

# Avoid division by zero if a feature has no variation
train_stds = train_stds.replace(0, 1)

# Apply training statistics to both datasets
X_train[numerical_columns] = (
    X_train[numerical_columns] - train_means
) / train_stds

X_test[numerical_columns] = (
    X_test[numerical_columns] - train_means
) / train_stds

print("\nFINAL TRAINING SET SHAPE")
print(X_train.shape)

print("\nFINAL TESTING SET SHAPE")
print(X_test.shape)

print("\nMISSING VALUES IN FINAL TRAINING SET")
print(X_train.isnull().sum().sum())

print("\nMISSING VALUES IN FINAL TESTING SET")
print(X_test.isnull().sum().sum())

print("\nFIRST FIVE PREPARED TRAINING ROWS")
print(X_train.head())

 #Save the prepared datasets for modeling 
import os

os.makedirs("processed_data", exist_ok=True)

X_train.to_csv("processed_data/X_train.csv", index=False)
X_test.to_csv("processed_data/X_test.csv", index=False)
y_train.to_csv("processed_data/y_train.csv", index=False)
y_test.to_csv("processed_data/y_test.csv", index=False)

print("\nPrepared datasets saved successfully,big boss!")
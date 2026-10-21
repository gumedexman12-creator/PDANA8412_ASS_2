
import pandas as pd

#Load dataset 
df = pd.read_csv("data/Books_Data_Clean.csv")

print("Dataset loaded successfully big!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

#Create target variable (number of book sales)
#The top 25% of books by units sold are classified as bestsellers.
sales_threshold = df["units sold"].quantile(0.75)

df["bestseller"] = (
    df["units sold"] >= sales_threshold
).astype(int)

print("\n***BESTSELLER SALES THRESHOLD***")
print(sales_threshold)

print("\n***BESTSELLER CLASS COUNTS***")
print(df["bestseller"].value_counts().sort_index())

print("\n***BESTSELLER CLASS PERCENTAGES***")
print(
    df["bestseller"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)
#Exploratory Data Analysis (EDA) - Part 1
#Inspect Dataset
print("\n***DATASET COLUMNS***")
print(df.columns.tolist())

print("\n***COLUMN DATA TYPES***")
print(df.dtypes)

print("\n***FIRST TEN ROWS***")
print(df.head(10))
print("\n***MISSING VALUES PER COLUMN***")
print(df.isnull().sum())

print("\nDataset inspection completed successfully, big boss!!!")

#Investigate dataset summary statistics and correlation with target variable
print("\n--- FINANCIAL AND SALES VARIABLES ---")

print(
    df[
        [
            "units sold",
            "gross sales",
            "publisher revenue",
            "sale price",
            "sales rank",
        ]
    ].describe().round(2)
)

print("\n--- CORRELATION WITH UNITS SOLD ---")

print(
    df[
        [
            "units sold",
            "gross sales",
            "publisher revenue",
            "sale price",
            "sales rank",
        ]
    ]
    .corr()
    .round(3)
)

#Exploratory Data Analysis (EDA) - Part 2 

# EDA Part 2: Compare predictors by bestseller status

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Numerical variables to investigate
numeric_features = [
    "Publishing Year",
    "Book_average_rating",
    "Book_ratings_count",
    "sale price"
]

# Categorical variables to investigate
categorical_features = [
    "genre",
    "language_code",
    "Author_Rating"
]

# Compare numerical variables by bestseller status
print("\nNUMERICAL FEATURES BY BESTSELLER STATUS")

for column in numeric_features:
    if column in df.columns:
        print(f"\n{column}")
        print(
            df.groupby("bestseller")[column]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(2)
        )

# Compare categorical variables by bestseller status
print("\nCATEGORICAL FEATURES BY BESTSELLER STATUS")

for column in categorical_features:
    if column in df.columns:
        print(f"\n{column}")
        print(
            pd.crosstab(
                df[column],
                df["bestseller"],
                normalize="index"
            ).round(3)
        )


# EDA Part 3: Data quality and category counts

print("PUBLISHING YEAR CHECK")
print("Years below 0:")
print(df.loc[df["Publishing Year"] < 0, "Publishing Year"].value_counts())

print("\nYears before 1450:")
print(df.loc[df["Publishing Year"] < 1450, "Publishing Year"].describe())

print("\nMissing values:")
print(df[[
    "Publishing Year",
    "Book_average_rating",
    "Book_ratings_count",
    "sale price"
]].isnull().sum())

print("\nCATEGORY COUNTS")

for column in ["genre", "language_code", "Author_Rating"]:
    if column in df.columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False))
        print("\nBestseller counts:")
        print(pd.crosstab(df[column], df["bestseller"]))

#EDA Part 4: Visualizations
import matplotlib.pyplot as plt
import pandas as pd

#One window containing five graphs
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

#Graph 1: Distribution of bestseller status

class_counts = df["bestseller"].value_counts().reindex(
    [0, 1], fill_value=0
)

axes[0].bar(
    ["Non-bestseller (0)", "Bestseller (1)"],
    class_counts.values
)
axes[0].set_title("Distribution of Bestseller Status")
axes[0].set_ylabel("Number of Books")
axes[0].tick_params(axis="x", rotation=15)

#Graph 2: Average rating by bestseller status

rating_groups = [
    df.loc[df["bestseller"] == 0, "Book_average_rating"].dropna(),
    df.loc[df["bestseller"] == 1, "Book_average_rating"].dropna()
]

axes[1].boxplot(
    rating_groups,
    tick_labels=["Non-bestseller", "Bestseller"]
)
axes[1].set_title("Average Rating by Bestseller Status")
axes[1].set_ylabel("Average Rating")

#Graph 3: Sale price by bestseller status

price_groups = [
    df.loc[df["bestseller"] == 0, "sale price"].dropna(),
    df.loc[df["bestseller"] == 1, "sale price"].dropna()
]

axes[2].boxplot(
    price_groups,
    tick_labels=["Non-bestseller", "Bestseller"]
)
axes[2].set_title("Sale Price by Bestseller Status")
axes[2].set_ylabel("Sale Price")

#Graph 4: Bestseller status by genre

genre_proportions = pd.crosstab(
    df["genre"],
    df["bestseller"],
    normalize="index"
).reindex(columns=[0, 1], fill_value=0) * 100

genre_proportions.plot(
    kind="bar",
    stacked=True,
    ax=axes[3]
)

axes[3].set_title("Bestseller Status by Genre")
axes[3].set_xlabel("Genre")
axes[3].set_ylabel("Percentage of Books")
axes[3].legend(
    ["Non-bestseller", "Bestseller"],
    title="Status"
)
axes[3].tick_params(axis="x", rotation=25)

#Graph 5: Bestseller status by author rating

author_proportions = pd.crosstab(
    df["Author_Rating"],
    df["bestseller"],
    normalize="index"
).reindex(columns=[0, 1], fill_value=0) * 100

author_proportions.plot(
    kind="bar",
    stacked=True,
    ax=axes[4]
)

axes[4].set_title("Bestseller Status by Author Rating")
axes[4].set_xlabel("Author Rating")
axes[4].set_ylabel("Percentage of Books")
axes[4].legend(
    ["Non-bestseller", "Bestseller"],
    title="Status"
)
axes[4].tick_params(axis="x", rotation=15)

# Hide the unused sixth panel (reference cha) 
axes[5].axis("off")

#Improve spacing and display everything together
fig.suptitle(
    "Exploratory Data Analysis: Bestseller Prediction",
    fontsize=16
)
plt.tight_layout()
plt.show()

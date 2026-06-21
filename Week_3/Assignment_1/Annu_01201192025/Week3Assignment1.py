# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# LOAD DATASET
df = pd.read_csv("agriculture.csv")

# ==========================
# Q1 DATASET OVERVIEW
# ==========================
print("Rows and Columns:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

# ==========================
# Q2 DATA TYPES & MISSING VALUES
# ==========================
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# Q3 DESCRIPTIVE STATISTICS
# ==========================
print("\nSummary Statistics:")
print(df.describe())

# ==========================
# Q4 HISTOGRAMS
# ==========================
columns = [
    "rainfall_mm",
    "temperature_c",
    "fertilizer_kg",
    "yield_ton_per_hectare"
]

for col in columns:
    plt.figure(figsize=(5,3))
    plt.hist(df[col], bins=10)
    plt.title(col)
    plt.show()

# ==========================
# Q5 CROP TYPE ANALYSIS
# ==========================
print("\nCrop Type Count:")
print(df["crop_type"].value_counts())

sns.countplot(x="crop_type", data=df)
plt.show()

# ==========================
# Q6 SOIL TYPE ANALYSIS
# ==========================
print("\nSoil Type Count:")
print(df["soil_type"].value_counts())

sns.countplot(x="soil_type", data=df)
plt.show()

# ==========================
# Q7 YIELD DISTRIBUTION
# ==========================
plt.hist(df["yield_ton_per_hectare"], bins=10)
plt.title("Yield Distribution")
plt.show()

# ==========================
# Q8 SCATTER PLOTS
# ==========================
plt.scatter(df["rainfall_mm"],
            df["yield_ton_per_hectare"])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.show()

plt.scatter(df["fertilizer_kg"],
            df["yield_ton_per_hectare"])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.show()

# ==========================
# Q9 CORRELATION MATRIX
# ==========================
corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.show()

# ==========================
# Q10 GROUP ANALYSIS
# ==========================
crop_yield = df.groupby("crop_type")["yield_ton_per_hectare"].mean()
print("\nAverage Yield by Crop:")
print(crop_yield)

soil_yield = df.groupby("soil_type")["yield_ton_per_hectare"].mean()
print("\nAverage Yield by Soil:")
print(soil_yield)

# ==========================
# Q11 ONE HOT ENCODING
# ==========================
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nFirst 5 Rows After Encoding:")
print(df_encoded.head())

# ==========================
# Q12 FEATURE SELECTION
# ==========================
X = df_encoded.drop("yield_ton_per_hectare", axis=1)
y = df_encoded["yield_ton_per_hectare"]

print("\nTarget Variable:")
print("yield_ton_per_hectare")

# ==========================
# Q13 TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nX_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# ==========================
# Q14 LINEAR REGRESSION
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coef_df)

highest_feature = coef_df.loc[
    coef_df["Coefficient"].idxmax()
]

print("\nHighest Positive Coefficient:")
print(highest_feature)
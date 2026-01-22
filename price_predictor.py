import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Save file path
file_path = "melb_data.csv"

# Read data and save in melb_dataset
melb_dataset = pd.read_csv(file_path)

# Features
X = melb_dataset.drop(['Price'], axis = 1)

# Target
y = melb_dataset.Price

# Train Test Split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# Select categorical columns and numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()
< 10 and X_train_full[cname].dtype == "object"]


my_cols = numerical_cols + categorical_cols


X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Preprocessing numerical data
numerical_transformer = SimpleImputer(strategy = 'median')

# Preprocessing Categorical data

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

# Build transformer safely
transformers = []
if numerical_cols:
    transformers.append(('num', numerical_transformer, numerical_cols))
if categorical_cols:
    transformers.append(('cat', categorical_transformer, categorical_cols))

# Bundle preprocessing for numerical and categorical data
preprocessors = ColumnTransformer(
    transformers=transformers, remainder='drop'
)
# Define Model
model = XGBRegressor(n_estimators = 1500, learning_rate = 0.02, n_jobs=5, objective="reg:squarederror", max_depth =6,random_state=0)

# Create and evaluate Pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessors),
    ('model', model)
])

# Fit model
my_pipeline.fit(X_train, y_train)

# Get Predictions
predictions = my_pipeline.predict(X_valid)

# Evaluate model
print("Mean Absolute Error: ", mean_absolute_error(y_valid, predictions))


# User Input
print("\n --- Enter house details to predict Price ---")
user_data = {}
for col in my_cols:
    if col in numerical_cols:
        # numeric input
        user_data[col] = float(input(f"Enter {col}: "))
    else:
        # categorical input
        user_data[col] = input(f"Enter {col}: ")

# Convert to dataframe
user_df = pd.DataFrame([user_data], columns=my_cols)

#Predict
predicted_price = my_pipeline.predict(user_df)[0]
# Output the price prediction
print("\nPredicted House Price: ", predicted_price)



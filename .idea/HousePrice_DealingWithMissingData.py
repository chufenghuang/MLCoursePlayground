import pandas as pd
from sklearn.model_selection import train_test_split
# house price file path
melbourne_file_path = "/Users/huangcf/IdeaProjects/KaggleMLCoursePlayground/.idea/melb_data.csv"

# read the data
data = pd.read_csv(melbourne_file_path)
# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10,random_state=0)
    model.fit(X_train,y_train)
    preds = model.predict(X_valid)
    return (mean_absolute_error(y_valid,preds))

#=====Drop columns that have missing value
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any() ]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train,reduced_X_valid,y_train,y_valid))

#=====Find missing value and replace it with mean value
from sklearn.impute import SimpleImputer
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train,imputed_X_valid,y_train,y_valid))

#=====Find missing value and replace it with mean value and add a column to tag which values were imputed.
# Make copy to avoid changing original data (when imputing)
X_train_clone = X_train.copy()
X_valid_clone = X_valid.copy()


# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_clone[col + "_was_missing"] = X_train_clone[col].isnull()
    X_valid_clone[col + "_was_missing"] = X_valid_clone[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_clone= pd.DataFrame(my_imputer.fit_transform(X_train_clone))
imputed_X_valid_clone = pd.DataFrame(my_imputer.transform(X_valid_clone))

# Imputation removed column names; put them back
imputed_X_train_clone.columns = X_train_clone.columns
imputed_X_valid_clone.columns = X_valid_clone.columns
print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_clone,imputed_X_valid_clone,y_train,y_valid))
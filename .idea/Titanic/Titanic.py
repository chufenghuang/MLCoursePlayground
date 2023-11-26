import pandas as pd
from sklearn.model_selection import train_test_split
#The trainning data file path
train_data_file_path = "/Users/huangcf/IdeaProjects/KaggleMLCoursePlayground/.idea/Titanic/train.csv"

train_data_full = pd.read_csv(train_data_file_path)
# print(train_data_full.head())

y = train_data_full.Survived
X = train_data_full.drop(['Survived'], axis=1)
#split the X,y data into train part and valid part.
X_train_full,X_valid_full,y_train,y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

#categorical columns
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns
                 if X_train_full[cname].dtype in ['int64','float64']]
#keep categorical columns and numerical columns
kept_cols = categorical_cols + numerical_cols
# print(kept_cols)
X_train = X_train_full[kept_cols].copy()
X_valid = X_valid_full[kept_cols].copy()



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#Pipeline Step 1: For the missing values, add the indicator column.
class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, suffix="_was_missing"):
        self.suffix = suffix

    def fit(self, X, y=None):
        # We need to find out which columns have missing values
        self.cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
        return self

    def transform(self, X):
        # For each column with missing data, add a binary column indicating missingness
        for col in self.cols_with_missing:
            X[col + self.suffix] = X[col].isnull()
        return X

#Pipeline Step 2: Replace the missing values of numerical columns with mean value, transfer categorical columns int to one hot format.
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numerical_transformer,numerical_cols),
        ('cat',categorical_transformer,categorical_cols)
    ]
)

#Pipeline Step 3: Define the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
model = XGBClassifier()

my_Pipeline = Pipeline(steps=[
    ('add_missing_indicators',MissingIndicator()),# Step 1: Add missing indicators
    ('preprocessor',preprocessor),#Step 2: deal with missing values and categorical values
    ('model',model)
])

# Preprocessing of training data, fit model
my_Pipeline.fit(X_train,y_train)
probabilities = my_Pipeline.predict_proba(X_valid)[:, 1]
threshold = 0.5  # This is a common threshold but can be adjusted based on your needs
predictions_binary = (probabilities >  threshold).astype(int)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Now you can use the probabilities to calculate the ROC AUC score
roc_auc = roc_auc_score(y_valid, probabilities)

# The rest of your metrics calculations can remain the same
accuracy = accuracy_score(y_valid, predictions_binary)
f1 = f1_score(y_valid, predictions_binary)
conf_matrix = confusion_matrix(y_valid, predictions_binary)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")



submission_data_path = "/Users/huangcf/IdeaProjects/KaggleMLCoursePlayground/.idea/Titanic/test.csv"
submission_data = pd.read_csv(submission_data_path)
result =  my_Pipeline.predict(submission_data)

submission_data["Survived"] = result

outputfile = submission_data[["PassengerId","Survived"]]
outputfile.to_csv("Titanic_out_put.csv",index=False)







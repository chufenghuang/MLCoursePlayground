import pandas as pd

# house price file path
melbourne_file_path = "/Users/huangcf/IdeaProjects/KaggleMLCoursePlayground/.idea/melb_data.csv"

# read the data
melbourne_house_data = pd.read_csv(melbourne_file_path)
melbourne_house_data = melbourne_house_data.dropna(axis=0)
print(melbourne_house_data.describe())
# the Price is the target we are gonna predict
y = melbourne_house_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_house_data[melbourne_features]

print(X.head())

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_Model = RandomForestRegressor(random_state=1)
forest_Model.fit(train_X,train_y)
model_preds = forest_Model.predict(val_X)
print(mean_absolute_error(val_y,model_preds))

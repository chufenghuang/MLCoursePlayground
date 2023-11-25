import pandas as pd

# house price file path
melbourne_file_path = "/Users/huangcf/IdeaProjects/KaggleMLCoursePlayground/.idea/melb_data.csv"

# read the data
melbourne_house_data = pd.read_csv(melbourne_file_path)

# show the basic info
# print(melbourne_house_data.describe())


melbourne_house_data = melbourne_house_data.dropna(axis=0)
print(melbourne_house_data.describe())
# the Price is the target we are gonna predict
y = melbourne_house_data.Price
print(y)

#define the features we want to use
melbourne_feature = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
X = melbourne_house_data[melbourne_feature]
print(X.describe())
print(X.head())

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

from sklearn.tree import DecisionTreeRegressor



from sklearn.metrics import mean_absolute_error

def get_MAE(max_leaf_nodes, train_X, val_X, train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    prediction = model.predict(val_X)
    mae_result = mean_absolute_error(val_y,prediction)
    return(mae_result)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in  [5, 50, 500, 5000]:
    this_mae = get_MAE(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: " , max_leaf_nodes , "  MAE is: " ,this_mae ,"\n")





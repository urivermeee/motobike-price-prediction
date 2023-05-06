import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

data=pd.read_excel('UsedMotorbikes.xlsx')
print(data.shape)

# Tạo các LabelEncoder cho từng cột
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_type = LabelEncoder()
le_location = LabelEncoder()
le_condition = LabelEncoder()
le_capacity = LabelEncoder()

# Mã hóa từng cột
data["Brand"] = le_brand.fit_transform(data["Brand"])
data["Model"] = le_model.fit_transform(data["Model"])
data["Type"] = le_type.fit_transform(data["Type"])
data["Location"] = le_location.fit_transform(data["Location"])
data["Condition"] = le_condition.fit_transform(data["Condition"])
data["Capacity"] = le_capacity.fit_transform(data["Capacity"])

X = data[['Brand', 'Model', 'Reg_Date', 'Km', 'Condition', 'Capacity', 'Type', 'Location']]
y = data['Price']

y=pd.DataFrame(y)
scaler = StandardScaler()
scaler.fit(y)
y = pd.DataFrame(scaler.transform(y),columns= y.columns )
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.035, random_state=42)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
import matplotlib.pyplot as plt
import seaborn as sns


##BAGGING REGRESSOR
regr = BaggingRegressor(base_estimator= DecisionTreeRegressor(),n_estimators=10, random_state=0).fit(X_train,y_train)
y_pred_ = regr.predict(X_train)

y_pred = regr.predict(X_test)

y_pre_regr_df = pd.DataFrame(y_pred, columns=['Price_predict'])
result_regr = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pre_regr_df], axis=1)



from sklearn.preprocessing import LabelEncoder
import joblib


# Lưu trữ các LabelEncoder
joblib.dump(le_brand, 'label_encoder_brand.pkl')
joblib.dump(le_model, 'label_encoder_model.pkl')
joblib.dump(le_type, 'label_encoder_type.pkl')
joblib.dump(le_location, 'label_encoder_location.pkl')
joblib.dump(le_condition, 'label_encoder_condition.pkl')
joblib.dump(le_capacity, 'label_encoder_capacity.pkl')

le__brand = joblib.load('label_encoder_brand.pkl')
le__model = joblib.load('label_encoder_model.pkl')
le__type = joblib.load('label_encoder_type.pkl')
le__location = joblib.load('label_encoder_location.pkl')
le__condition = joblib.load('label_encoder_condition.pkl')
le__capacity = joblib.load('label_encoder_capacity.pkl')

# Save model to file
joblib.dump(regr, 'Bagging Regressor.pkl')



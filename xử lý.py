from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import locale
locale.setlocale(locale.LC_ALL, 'vi_VN')

data=pd.read_excel('UsedMotorbikes.xlsx')

X = data[['Brand', 'Model', 'Reg_Date', 'Km', 'Condition', 'Capacity', 'Type', 'Location']]
y = data['Price']
y=pd.DataFrame(y)
scaler = StandardScaler()
scaler.fit(y)
y = pd.DataFrame(scaler.transform(y),columns= y.columns )


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    brand = request.form['brand']
    model = request.form['model']
    reg_date = request.form['reg_date']
    km = request.form['km']
    condition = request.form['condition']
    capacity = request.form['capacity']
    type = request.form['type']
    location = request.form['location']

    le__brand = joblib.load('label_encoder_brand.pkl')
    le__model = joblib.load('label_encoder_model.pkl')
    le__type = joblib.load('label_encoder_type.pkl')
    le__location = joblib.load('label_encoder_location.pkl')
    le__condition = joblib.load('label_encoder_condition.pkl')
    le__capacity = joblib.load('label_encoder_capacity.pkl')

    brand_encoded = le__brand.transform([brand])[0]
    model_encoded = le__model.transform([model])[0]
    condition_encoded = le__condition.transform([condition])[0]
    capacity_encoded = le__capacity.transform([capacity])[0]
    type_encoded = le__type.transform([type])[0]
    location_encoded = le__location.transform([location])[0]

    # Tạo DataFrame từ thông tin nhập vào
    input_df = pd.DataFrame({'Brand': [brand_encoded], 'Model': [model_encoded], 'Reg_Date': [reg_date], 'Km': [km],
                                     'Condition': [condition_encoded], 'Capacity': [capacity_encoded], 'Type': [type_encoded], 'Location': [location_encoded]})

    # Đưa dữ liệu vào model để dự đoán giá xe

    regr_model = joblib.load('Bagging Regressor.pkl')
    predicted_price = regr_model.predict(input_df)[0]
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1)).ravel()
    predicted_price = locale.currency(predicted_price, grouping=True, symbol=False)[:-3] + "VNĐ"
    # Hiển thị kết quả dự đoán giá xe lên trang web
    return render_template('result.html', predicted_price=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)


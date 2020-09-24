import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('reg_sel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    '''
    df_data = pd.read_csv('house_test_data.csv')
    sel_columns = [  'SqFtTotal',
                     'Bath',
                     'Bed',
                     'PropertyType',
                     'CondoType',
                     'Basement_Full',
                     'city',
                     'Community',  
                     'Basement_WalkoutWalkUp',
                     'Parking_DoubleGarageAttached',
                     'Parking_NoGarage']
    df_data = df_data[sel_columns]
    '''
    prediction = model.predict(final_features)

    print(prediction)
    prediction = scaler.inverse_transform(prediction)  #[scaler.inverse_transform(t[0]) for t in prediction]
    print(prediction)

    output = ', '.join(str(round(p[0],2)) for p in prediction)   #round(prediction[1], 2)

    print(output)

    return render_template('index.html', prediction_text='Home Price should be ${}'.format(output))


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host = '0.0.0.0',port=8080)
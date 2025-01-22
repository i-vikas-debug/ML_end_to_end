import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


model = pickle.load(open('models/regressor.pkl','rb'))
scal = pickle.load(open('models/scaler.pkl','rb'))

application = Flask(__name__)

app=application

# @app.route("/")

# def index():
#     return render_template('index.html')

@app.route('/',methods=['GET','POST'])

def predict():
    if request.method =='POST':
        MedInc = float(request.form.get('MedInc'))
        HouseAge = float(request.form.get('HouseAge'))
        AveRooms = float(request.form.get('AveRooms'))
        AveBedrms = float(request.form.get('AveBedrms'))
        Population = float(request.form.get('Population'))
        AveOccup = float(request.form.get('AveOccup'))
        Latitude = float(request.form.get('Latitude'))
        Longitude = float(request.form.get('Longitude'))

        new_scaled_data = scal.transform([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])

        result = model.predict(new_scaled_data)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run()
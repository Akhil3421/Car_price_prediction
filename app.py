import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
#load the model
model=pickle.load(open('rf.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
encoderr=pickle.load(open('encoder.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    new_data=stdd(data)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

def stdd(data):
    column_names = ['company', 'model','symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'fueltype',
       'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
       'enginetype', 'cylindernumber', 'fuelsystem']
    df = pd.DataFrame([data], columns=column_names)
    nums = df.drop(columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
       'enginetype', 'fuelsystem', 'company', 'model'])
    nums= nums.astype(float)
    scaler_columns=['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth',
       'carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
       'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'highwaympg']
    nums= nums[scaler_columns].copy()
    nums.columns = scaler_columns
    new_cat=pd.DataFrame(encoderr.transform([[df['fueltype'][0], df['aspiration'][0], df['carbody'][0],
     df['drivewheel'][0],df['enginelocation'][0], df['enginetype'][0], 
     df['fuelsystem'][0], df['company'][0], df['model'][0]]]),columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
       'enginetype', 'fuelsystem', 'company', 'model'])
    new_cat=new_cat.astype(float)
    new_nums=scaler.transform(nums)
    new_columns = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth',
       'carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
       'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'highwaympg', 'fueltype', 'aspiration', 'carbody', 'drivewheel',
       'enginelocation', 'enginetype', 'fuelsystem', 'company', 'model']
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums,columns=scaler_columns)],axis=1)
    new_data = new_data[new_columns].copy()
    return new_data


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    final_input=stdd(data)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Car predicted price is {}".format(output))



if __name__=="__main__":
    app.run()
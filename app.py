from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

with open ('house_price_prediction.pkl','rb') as f:
    model=pickle.load(f)

#define routes for class
#when you run flask app this is the first thing you are going to see
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[
        #all column names are called features
        float(request.form['CRI']),
        float(request.form['ZN']),
        float(request.form['INDUS']),
        float(request.form['CHAS']),
        float(request.form['NOX']),
        float(request.form['RM']),
        float(request.form['AGE']),
        float(request.form['DIS']),
        float(request.form['RAD']),
        float(request.form['TAX']),
        float(request.form['PTRATIO']),
        float(request.form['B']),
        float(request.form['LSTAT'])

    ]
    #inputs that we put in index.html is requested through features
    #rquest=requested values all these columns

    #converting all the columns into array using numpy

    features_array=np.array([features])

    prediction=model.predict(features_array)
    output=round(prediction[0],2)

    return render_template('index.html',prediction_text=f"Predicted Price: {output}")


if __name__=="__main__":
    app.run(debug=True)
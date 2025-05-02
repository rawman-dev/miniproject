import numpy as np
from flask import Flask,request,render_template
import pickle
fapp=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
@fapp.route("/")
def Home():
    return render_template("index.html")
@fapp.route("/predict",methods=["POST"])
def predict():
    gender_text = request.form.get('gender')
    gender = 0 if gender_text == 'Male' else 1
    age = float(request.form['age'])
    height = float(request.form.get('height'))
    weight = float(request.form['weight'])
    bmi= float(request.form.get('bmiw'))
    hl=float(request.form['hl'])
    he=float(request.form['he'])
    hs=float(request.form['hs'])
    chat_text = request.form.get('chat')
    chatg = 0 if chat_text == 'No' else 1
    features=np.array([[gender,age,height,weight,bmi,hl,he,hs,chatg]])
    prediction=model.predict(features)
    return render_template("index.html",prediction_text="The predicted GPA is {}".format(prediction))
if __name__=="__main__":
    fapp.run(debug=True)
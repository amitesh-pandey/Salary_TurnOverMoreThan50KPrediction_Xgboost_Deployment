# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 19:20:47 2021

@author: Amitesh
"""


from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

filename = 'final_model_xgboost.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        Age = request.form['age']
        Workclass = request.form['workclass']
        Fnlwgt = request.form['fnlwgt']
        Education = request.form['education']
        Education_num = request.form['education_num']
        Marital_status = request.form['marital_status']
        Occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        Sex = request.form['sex']
        Capital_gain = request.form['capital_gain']
        Capital_loss = request.form['capital_loss']
        Hours_per_week = request.form['hours_per_week']
        result = np.array([[Age, Workclass, Fnlwgt, Education, Education_num, Marital_status,
                            Occupation, relationship, race, Sex,
                            Capital_gain, Capital_loss, Hours_per_week]])
        prediction = model.predict(result)

        if prediction > 15:
            display = "TurnOver More than 50K"
        else:
            display = "TurnOver Less than 50K"
            print(display)

    return render_template("submit.html", n=display)


if __name__ == "__main__":
    app.run(debug = True)







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:06:13 2019

@author: shyam
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from flask import Flask, request, url_for, render_template



admis=pd.read_csv(r"C:\Users\vinay\Desktop\new\Admission_Predict.csv")

admis_train, admis_test = train_test_split(admis, test_size=0.20)

admis_train_1 = admis_train[['GRE Score','TOEFL Score','University Rating','CGPA','Research']].copy()

admis_test_1 = admis_test[['GRE Score','TOEFL Score','University Rating','CGPA','Research']].copy()

target=admis_train['Chance of Admit ']

lin_reg=LinearRegression()
lin_reg.fit(admis_train_1,target)
x=lin_reg.predict(admis_test_1)
y=admis_test['Chance of Admit ']

ms=mean_squared_error(x,y)


app =  Flask(__name__)
@app.route('/')
def index():
    return render_template('admission.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method=="POST":
        gre=request.form['Gre']
        toefl=request.form['Toefl']
        university=request.form['University']
        cgpa=request.form['CGPA']
        print(gre)
        print(university)
        print(toefl)
        print(cgpa)
        pre=[float(gre),float(toefl),float(university),float(cgpa),0]
        res=lin_reg.predict([pre])
        print(res)
        res=res*100
        print(res)
        return render_template('admission.html',result=int(res))
    return render_template('admission.html')

if __name__ == '__main__':
    app.run(debug = True)

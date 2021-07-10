from django.shortcuts import render

import pickle

import numpy as np



def home(request):

    return render(request, 'index.html')


def getPredictions(Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income):

    nbc = pickle.load(open('model.sav', 'rb'))

    scaled = pickle.load(open('scaler.sav', 'rb'))


    prediction = nbc.predict(scaled.transform([

        [Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income]

    ]))
    

    if prediction == 0:
        return 'no'

    elif prediction == 1:

        return 'yes'

    else:
        return 'error'

def score(Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income):

    nbc = pickle.load(open('model.sav', 'rb'))

    scaled = pickle.load(open('scaler.sav', 'rb'))


    prob1 = nbc.predict_proba(scaled.transform([

        [Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income]

    ]))

    prob = np.max(prob1)

    return prob

def result(request):

    Gender = int(request.GET['Gender'])

    Married = int(request.GET['Married'])

    Dependents = int(request.GET['Dependents'])

    Education = int(request.GET['Education'])

    Emp = int(request.GET['Self_Employed'])

    Credit_History = int(request.GET['Credit_History'])

    Property_Area = int(request.GET['Property_Area'])

    Total_Income = int(request.GET['Total_Income'])
    

    result = getPredictions(Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income)
    conf_score = round(np.max(score(Gender,Married,Dependents,Education,Emp,Credit_History,Property_Area,Total_Income)*100),2)

    return render(request, 'result.html', {'result':result,'conf_score':conf_score} )

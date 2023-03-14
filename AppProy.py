#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:27:39 2023
@author: julianacepeda
"""

import dash
from dash import dcc  # dash core components
from dash import html # dash html components

from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator

import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

model = BayesianNetwork([('thal','trestbps'),('age','trestbps'),('age','chol'),('sex','chol'),('slope','num'),('trestbps','num'),('chol','num'),('fbs','num'),('restecg','num'),('exang','num'),('num','thalach'),('num','cp')])
df = pd.read_csv('datosFinal')
emv = MaximumLikelihoodEstimator(model=model, data=df)
model.fit(data=df, estimator = MaximumLikelihoodEstimator) 
eby = BayesianEstimator(model=model, data=df)
inference = VariableElimination(model)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

options = [
    {'label': 'Colesterol', 'value': 'chol'},
    {'label': 'Azúcar', 'value': 'fbs'},
    {'label': 'Diagnóstico', 'value': 'num'},
    {'label': 'Talesemia', 'value': 'thal'},
    {'label': 'Angina', 'value': 'exang'}
]

binario = [{'label': '0','value': 0},
           {'label': '1', 'value': 1}]

app.layout = html.Div(children=[
    html.H1(children='Paciente', style={'font-size': '3rem'}),
    html.Div('Tener en cuenta lo siguiente y digite el numero correspondiente en la casilla.'),
    html.Div('Edad: 0 para menores de 45 años, 1 para mayores de 45 años'),
    html.Div('Sexo: 0 para femenino, 1 para masculino'),
    html.Div('Presión arterial en reposo: 0 para menos de 120mm Hg, 1 para mas de 120mm Hg'),
    html.Div('Colesterol sérico: 0 para menos de 240mg/dL , 1 para mas de 240mg/dL'),
    html.Div('Glucemia en ayunas: 0 para menos de 120mg/dl, 1 para mas de 120mg/dL'),
    html.Div('Talasemia: 0 cuando sea normal (3), 1 cuando sea anormal (6 y 7)'),
    html.Div('Resultados electrocardiográficos en reposo: 0 para normal (nivel 0), 1 para anormalidades (nivel 1 y 2)'),
    html.Div('Frecuencia cardíaca máxima alcanzada: 0 para menos de 220-edad, 1 para mas de 220-edad'),
    html.Div('Angina inducida por el ejercicio: 0 si no fue inducida por estrés, 1 si si fue inducida por estrés'),
    html.Div('Pendiente del segmento ST de ejercicio máximo: 0 si se mantiene plana o sube, 1 si desciende'),
    html.Div('Dolor de pecho: 0 si es asintomatica, 1 si siente dolor '),
    
    html.Div([
        html.Label('Edad:'),
        dcc.Dropdown(id='dropdown-age',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Sexo:'),
        dcc.Dropdown(id='input-sex',options=binario,
        value=None,
        style = {'width': '100px'})
        ],style={'display': 'inline-block', 'margin-right': '200px'}),
    
    html.Div([
        html.Label('Presión arterial:'),
        dcc.Dropdown(id='input-trestbps',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Colesterol sérico:'),
        dcc.Dropdown(id='input-chol',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Glucemia en ayunas:'),
        dcc.Dropdown(id='input-fbs',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Talasemia:'),
        dcc.Dropdown(id='input-thal',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Resultados electrocardiográficos en reposo:'),
        dcc.Dropdown(id='input-restecg',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Frecuencia cardíaca máxima alcanzada:'),
        dcc.Dropdown(id='input-thalach',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Angina inducida por el ejercicio:'),
        dcc.Dropdown(id='input-exang',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Pendiente del segmento ST de ejercicio máximo:'),
        dcc.Dropdown(id='input-slope',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.Div([
        html.Label('Dolor de pecho:'),
        dcc.Dropdown(id='input-cp',options=binario,
        value=None,
        style = {'width': '100px'})
        ]),
    
    html.H6('Selecciona una opción a determinar'),
    dcc.Dropdown(
        id='dropdown-options',
        options=options,
        value=None,
        style = {'width': '200px'}
        ),
    html.Button('Guardar', id='boton-guardar'),
    html.Div(id='resultado')
    ]
)

             
@app.callback(
    dash.dependencies.Output('resultado', 'children'),
    
    [dash.dependencies.Input('boton-guardar', 'n_clicks')],
    [dash.dependencies.State('dropdown-age', 'value'),
     dash.dependencies.State('input-sex', 'value'),
     dash.dependencies.State('input-trestbps', 'value'),
     dash.dependencies.State('input-chol', 'value'),
     dash.dependencies.State('input-fbs', 'value'),
     dash.dependencies.State('input-thal', 'value'),
     dash.dependencies.State('input-restecg', 'value'),
     dash.dependencies.State('input-thalach', 'value'),
     dash.dependencies.State('input-exang', 'value'),
     dash.dependencies.State('input-slope', 'value'),
     dash.dependencies.State('input-cp', 'value'),
     dash.dependencies.State('dropdown-options', 'value')]
)

def guardar_datos(n_clicks, age, sex, trestbps, chol, fbs, thal,restecg, thalach, exang, slope, cp, value ):
    if n_clicks is None:
        return ''
    else:
       lista = {"age": age, "sex": sex, "trestbps": trestbps, "chol": chol, "fbs": fbs, "thal": thal, "restecg": restecg, "thalach": thalach, "exang": exang, "slope": slope, "cp": cp}
       lista = {k: v for k, v in lista.items() if v is not None}

       query = inference.query(variables = [value], evidence = lista)
       
       if value == 'chol':
           return f'El paciente tiene una probabilidad de {query.values[1]*100:,.2f}% de tener colesterol alto.'
       if value == 'num':
           return f'El paciente tiene una probabilidad de {query.values[1]*100:,.2f}% de padecer una enfermedad cardíaca.'
       if value == 'fbs':
           return f'El paciente tiene una probabilidad de {query.values[1]*100:,.2f}% de tener niveles de azucar altos.'
       if value == 'thal':
           return f'El paciente tiene una probabilidad de {query.values[1]*100:,.2f}% de padecer Talasemia'
       if value == 'exang':
           return f'El paciente tiene una probabilidad de {query.values[1]*100:,.2f}% de que el dolor sea causado por ejercicio'
       if value == None:
           return ''

if __name__ == '__main__':
    app.run_server(debug=True)
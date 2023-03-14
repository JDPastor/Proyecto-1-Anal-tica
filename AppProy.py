#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:27:39 2023

@author: julianacepeda
"""


import dash
from dash import dcc  # dash core components
from dash import html # dash html components


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

options = [
    {'label': 'Colesterol', 'value': 'colesterol'},
    {'label': 'Azúcar', 'value': 'azucar'},
    {'label': 'Diagnóstico', 'value': 'diagnostico'},
    {'label': 'Talesemia', 'value': 'talesemia'},
    {'label': 'Angina', 'value': 'angina'}
]

app.layout = html.Div(children=[
    html.H1(children='Paciente'),
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
        dcc.Input(id='input-age', type='number', value=''),
    ]),
    html.Div([
        html.Label('Sexo:'),
        dcc.Input(id='input-sex', type='number', value=''),
    ]),
    
    
    html.Div([
        html.Label('Presión arterial en reposo:'),
        dcc.Input(id='input-trestbps', type='number', value=''),
    ]),
    html.Div([
        html.Label('Colesterol sérico:'),
        dcc.Input(id='input-chol', type='number', value=''),
    ]),
    html.Div([
        html.Label('Glucemia en ayunas:'),
        dcc.Input(id='input-fbs', type='number', value=''),
    ]),
    html.Div([
        html.Label('Talasemia'),
        dcc.Input(id='input-thal', type='number', value=''),
    ]),
    html.Div([
        html.Label('Resultados electrocardiográficos en reposo:'),
        dcc.Input(id='input-restecg', type='number', value=''),
    ]),
    html.Div([
        html.Label('Frecuencia cardíaca máxima alcanzada:'),
        dcc.Input(id='input-thalach', type='number', value=''),
    ]),
    html.Div([
        html.Label('Angina inducida por el ejercicio:'),
        dcc.Input(id='input-exang', type='number', value=''),
    ]),
    html.Div([
        html.Label('Pendiente del segmento ST de ejercicio máximo:'),
        dcc.Input(id='input-slope', type='number', value=''),
    ]),
    html.Div([
        html.Label('Dolor de pecho:'),
        dcc.Input(id='input-cp', type='number', value=''),
    ]),
    
    
    
    html.H6('Selecciona una opción a determinar'),
    dcc.Dropdown(
        id='dropdown-options',
        options=options,
        value=options[0]['value']
    ),
    
    html.Button('Guardar', id='boton-guardar'),
    html.Div(id='resultado')

    ]
)


             
@app.callback(
    dash.dependencies.Output('resultado', 'children'),
    
    [dash.dependencies.Input('boton-guardar', 'n_clicks')],
    [dash.dependencies.State('input-age', 'value'),
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
        return f'La edad ingresada es: {age} y el sexo es: {sex}'


if __name__ == '__main__':
    app.run_server(debug=True)
    
    
  
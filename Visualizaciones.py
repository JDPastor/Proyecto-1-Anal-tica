#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:03:45 2023

@author: julianacepeda
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

model = BayesianNetwork([('thal','trestbps'),('age','trestbps'),('age','chol'),('sex','chol'),('slope','num'),('trestbps','num'),('chol','num'),('fbs','num'),('restecg','num'),('exang','num'),('num','thalach'),('num','cp')])
df = pd.read_csv('datosNoNA')
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

num= df['num']
conteo1 = df['num'].value_counts()
conteo2 = df['sex'].value_counts()
conteo3 = df['age'].value_counts()
conteo4 = df['thal'].value_counts()


df_g1 = df.groupby(['num','sex']).agg({'num': 'count'})
vectH = df_g1.loc[df_g1['num'] != 1]
vectM = []

conteo5 = df['num'].value_counts()

trace1 = go.Bar(x=df['num'], y=df['sex'], name='Variable 1')
trace2 = go.Bar(x=df['num'], y=df['sex'], name='Variable 2')
union = [trace1, trace2]

layout = go.Layout(
    barmode='group',
    title='Comparación de variables por categoría',
    xaxis=dict(title='Categoría'),
    yaxis=dict(title='Valor')
)

figure = go.Figure(data=union, layout=layout)

app.layout = html.Div(children=[
    html.H1(children='Título del Tablero'),

    dcc.Graph(
        id='grafico',
        figure=px.bar(x=conteo1.index, y=conteo1.values, title='Título del Gráfico')
    ),

    html.P(children='Descripción del Gráfico', style={'font-size': '16px'})
])


app.layout = html.Div(children=[
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Vizualizacion', value='tab-1', children=[
            html.H1(children='Pestaña 1 - Visualizaciones', style={'text-align': 'center', 'font-weight': 'bold', 'color': 'darkblue'}),
            dcc.Graph(
                id='grafico',
                figure=px.bar(x=conteo1.index, y=conteo1.values, title='Diagnósticos de enfermedad cardiaca', )
                    .update_layout(xaxis_title="Diagnóstico", yaxis_title="Número de pacientes")
            ),
            html.P(children='0 si no tiene enfermedad cardiaca y 1 a 4 si tiene enfermedad dependiendo de su gravedad', style={'text-align': 'center','font-size': '16px'}),
            dcc.Graph(
                id='grafico2',
                figure=px.bar(x=conteo5.index, y=conteo5.values, title='Sexo pacientes' )
            ),
            html.P(children='0 (rojo) si es mujer y 1 (azul) si es hombre', style={'text-align': 'center','font-size': '16px'}),
            dcc.Graph(
                id='grafico3',
                figure=px.bar(x=conteo3.index, y=conteo3.values, title='Edades pacientes')
                    .update_layout(xaxis_title="Edades", yaxis_title="Número de pacientes")
            ),
            html.P(children='Muestra la distribucion de las edades de los pacientes', style={'text-align': 'center','font-size': '16px'}),
            dcc.Graph(
                id='grafico4',
                figure=px.pie(names=conteo4.index, values=conteo4.values, title='Clasificación para la enfermedad talasemia')
            ),
            html.P(children='Clasificación para la enfermedad talasemia; 3 si no sufre de la condición, 6 o 7 si sufre de la condición', style={'text-align': 'center', 'font-size': '16px'}),
        ]),
        dcc.Tab(label='Modelo Antiguo', value='tab-2', children=[
            html.H1(children='Pestaña 2 - Modelo Antiguo', style={'text-align': 'center', 'font-weight': 'bold', 'color': 'darkblue'}),
            html.H1(children='Paciente', style={'font-size': '30px', 'font-weight': 'bold', 'background-color': 'lightblue', 'padding': '10px'}),
            html.Div('Tener en cuenta lo siguiente y digite el numero correspondiente en la casilla.', style={'font-size':'18pt'}),
            html.Div('-Edad: 0 para menores de 45 años, 1 para mayores de 45 años'),
            html.Div('-Sexo: 0 para femenino, 1 para masculino'),
            html.Div('-Presión arterial en reposo: 0 para menos de 120mm Hg, 1 para mas de 120mm Hg'),
            html.Div('-Colesterol sérico: 0 para menos de 240mg/dL , 1 para mas de 240mg/dL'),
            html.Div('-Glucemia en ayunas: 0 para menos de 120mg/dl, 1 para mas de 120mg/dL'),
            html.Div('-Talasemia: 0 cuando sea normal (3), 1 cuando sea anormal (6 y 7)'),
            html.Div('-Resultados electrocardiográficos en reposo: 0 para normal (nivel 0), 1 para anormalidades (nivel 1 y 2)'),
            html.Div('-Frecuencia cardíaca máxima alcanzada: 0 para menos de 220-edad, 1 para mas de 220-edad'),
            html.Div('-Angina inducida por el ejercicio: 0 si no fue inducida por estrés, 1 si si fue inducida por estrés'),
            html.Div('-Pendiente del segmento ST de ejercicio máximo: 0 si se mantiene plana o sube, 1 si desciende'),
            html.Div('-Dolor de pecho: 0 si es asintomatica, 1 si siente dolor '),
            html.H1(children=' ', style={ 'background-color': 'lightblue', 'padding': '10px'}),
            
            html.Div([
                html.Label('Edad:'),
                dcc.Dropdown(id='dropdown-age',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Sexo:'),
                dcc.Dropdown(id='input-sex',options=binario,
                value=None,
                style = {'width': '100px'})
            ],style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Presión arterial:'),
                dcc.Dropdown(id='input-trestbps',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Colesterol sérico:'),
                dcc.Dropdown(id='input-chol',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Glucemia en ayunas:'),
                dcc.Dropdown(id='input-fbs',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Talasemia:'),
                dcc.Dropdown(id='input-thal',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Resultados electrocardiográficos en reposo:'),
                dcc.Dropdown(id='input-restecg',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Frecuencia cardíaca máxima alcanzada:'),
                dcc.Dropdown(id='input-thalach',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Angina inducida por el ejercicio:'),
                dcc.Dropdown(id='input-exang',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Pendiente del segmento ST de ejercicio máximo:'),
                dcc.Dropdown(id='input-slope',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Dolor de pecho:'),
                dcc.Dropdown(id='input-cp',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            html.H1(children=' ', style={ 'background-color': 'white', 'padding': '10px'}),
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
            ),
        dcc.Tab(label='Modelo Nuevo', value='tab-3', children=[
            html.H1(children='Pestaña 3 - Modelo Nuevo', style={'text-align': 'center', 'font-weight': 'bold', 'color': 'darkblue'}),
            html.H1(children='Paciente', style={'font-size': '30px', 'font-weight': 'bold', 'background-color': 'lightblue', 'padding': '10px'}),
            html.Div('Tener en cuenta lo siguiente y digite el numero correspondiente en la casilla.', style={'font-size':'18pt'}),
            html.Div('-Edad: 0 para menores de 45 años, 1 para mayores de 45 años'),
            html.Div('-Sexo: 0 para femenino, 1 para masculino'),
            html.Div('-Presión arterial en reposo: 0 para menos de 120mm Hg, 1 para mas de 120mm Hg'),
            html.Div('-Colesterol sérico: 0 para menos de 240mg/dL , 1 para mas de 240mg/dL'),
            html.Div('-Glucemia en ayunas: 0 para menos de 120mg/dl, 1 para mas de 120mg/dL'),
            html.Div('-Talasemia: 0 cuando sea normal (3), 1 cuando sea anormal (6 y 7)'),
            html.Div('-Resultados electrocardiográficos en reposo: 0 para normal (nivel 0), 1 para anormalidades (nivel 1 y 2)'),
            html.Div('-Frecuencia cardíaca máxima alcanzada: 0 para menos de 220-edad, 1 para mas de 220-edad'),
            html.Div('-Angina inducida por el ejercicio: 0 si no fue inducida por estrés, 1 si si fue inducida por estrés'),
            html.Div('-Pendiente del segmento ST de ejercicio máximo: 0 si se mantiene plana o sube, 1 si desciende'),
            html.Div('-Dolor de pecho: 0 si es asintomatica, 1 si siente dolor '),
            html.H1(children=' ', style={ 'background-color': 'lightblue', 'padding': '10px'}),
            
            html.Div([
                html.Label('Edad:'),
                dcc.Dropdown(id='dropdown-age',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Sexo:'),
                dcc.Dropdown(id='input-sex',options=binario,
                value=None,
                style = {'width': '100px'})
            ],style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Presión arterial:'),
                dcc.Dropdown(id='input-trestbps',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Colesterol sérico:'),
                dcc.Dropdown(id='input-chol',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Glucemia en ayunas:'),
                dcc.Dropdown(id='input-fbs',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Talasemia:'),
                dcc.Dropdown(id='input-thal',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Resultados electrocardiográficos en reposo:'),
                dcc.Dropdown(id='input-restecg',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Frecuencia cardíaca máxima alcanzada:'),
                dcc.Dropdown(id='input-thalach',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Angina inducida por el ejercicio:'),
                dcc.Dropdown(id='input-exang',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Pendiente del segmento ST de ejercicio máximo:'),
                dcc.Dropdown(id='input-slope',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Dolor de pecho:'),
                dcc.Dropdown(id='input-cp',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            html.H1(children=' ', style={ 'background-color': 'white', 'padding': '10px'}),
            html.H6('Selecciona una opción a determinar'),
            dcc.Dropdown(
                id='dropdown-options',
                options=options,
                value=None,
                style = {'width': '200px'}
                ),
            html.Button('Guardar', id='boton-guardar'),
            html.Div(id='resultado')
        
        ]),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

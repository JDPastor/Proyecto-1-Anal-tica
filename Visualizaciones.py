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
import pickle

from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('modelNuevo.pkl', 'rb') as f:
    model2 = pickle.load(f)

df = pd.read_csv('datosNoNA')

inference = VariableElimination(model)
inference2 = VariableElimination(model2)

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

df_female = df_g1.loc[df_g1.index.get_level_values('sex') == 0]
df_male = df_g1.loc[df_g1.index.get_level_values('sex') == 1]
count_female = df_female['num'].values
count_male = df_male['num'].values

num = {0,1,2,3,4}

bars = []

for num in num:
    # Select the rows for the current value of "num"
    df_num = df_g1.loc[num]
    # Create the two bar charts for the current value of "num"
    bar_female = go.Bar(
        x=[num],
        y=[df_num.loc[df_num.index.get_level_values('sex') == 0, 'num'].values[0]],
        name='Female',
        marker=dict(color='red'),
        width=0.4
    )
    bar_male = go.Bar(
        x=[num],
        y=[df_num.loc[df_num.index.get_level_values('sex') == 1, 'num'].values[0]],
        name='Male',
        marker=dict(color='blue'),
        width=0.4
    )
    bars.append(bar_female)
    bars.append(bar_male)

conteo5 = df['num'].value_counts()

union = [df_female, df_male]

layout = go.Layout(
    title='Diagnóstico por genero',
    barmode='group',
    showlegend = False,
    bargap = 0
)

fig = go.Figure(data=bars, layout=layout)

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
                figure=fig
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
                dcc.Dropdown(id='dropdown-age2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Sexo:'),
                dcc.Dropdown(id='input-sex2',options=binario,
                value=None,
                style = {'width': '100px'})
            ],style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Presión arterial:'),
                dcc.Dropdown(id='input-trestbps2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Colesterol sérico:'),
                dcc.Dropdown(id='input-chol2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Glucemia en ayunas:'),
                dcc.Dropdown(id='input-fbs2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Talasemia:'),
                dcc.Dropdown(id='input-thal2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Resultados electrocardiográficos en reposo:'),
                dcc.Dropdown(id='input-restecg2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Frecuencia cardíaca máxima alcanzada:'),
                dcc.Dropdown(id='input-thalach2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Angina inducida por el ejercicio:'),
                dcc.Dropdown(id='input-exang2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            
            html.Div([
                html.Label('Pendiente del segmento ST de ejercicio máximo:'),
                dcc.Dropdown(id='input-slope2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '50%'}),
            
            html.Div([
                html.Label('Dolor de pecho:'),
                dcc.Dropdown(id='input-cp2',options=binario,
                value=None,
                style = {'width': '100px'})
            ], style={'display': 'inline-block', 'width': '500px'}),
            html.H1(children=' ', style={ 'background-color': 'white', 'padding': '10px'}),
            html.H6('Selecciona una opción a determinar'),
            dcc.Dropdown(
                id='dropdown-options2',
                options=options,
                value=None,
                style = {'width': '200px'}
                ),
            html.Button('Guardar', id='boton-guardar2'),
            html.Div(id='resultado2')
        
        ]),
    ])
])

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

@app.callback(
    dash.dependencies.Output('resultado2', 'children'),
    
    [dash.dependencies.Input('boton-guardar2', 'n_clicks')],
    [dash.dependencies.State('dropdown-age2', 'value'),
     dash.dependencies.State('input-sex2', 'value'),
     dash.dependencies.State('input-trestbps2', 'value'),
     dash.dependencies.State('input-chol2', 'value'),
     dash.dependencies.State('input-fbs2', 'value'),
     dash.dependencies.State('input-thal2', 'value'),
     dash.dependencies.State('input-restecg2', 'value'),
     dash.dependencies.State('input-thalach2', 'value'),
     dash.dependencies.State('input-exang2', 'value'),
     dash.dependencies.State('input-slope2', 'value'),
     dash.dependencies.State('input-cp2', 'value'),
     dash.dependencies.State('dropdown-options2', 'value')]
)

def guardar_datos(n_clicks, age, sex, trestbps, chol, fbs, thal,restecg, thalach, exang, slope, cp, value ):
    if n_clicks is None:
        return ''
    else:
       lista = {"age": age, "sex": sex, "trestbps": trestbps, "chol": chol, "fbs": fbs, "thal": thal, "restecg": restecg, "thalach": thalach, "exang": exang, "slope": slope, "cp": cp}
       lista = {k: v for k, v in lista.items() if v is not None}

       query = inference2.query(variables = [value], evidence = lista)
       
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

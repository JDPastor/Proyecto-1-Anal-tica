#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:26:05 2023

@author: julianacepeda
"""

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import pandas as pd
import pickle
import xgboost as xgb

df = pd.read_csv('DatosP3.csv')
df.dropna(inplace=True)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

conteo = df.groupby(['COLE_NATURALEZA', 'COLE_BILINGUE']).size().reset_index(name='Count')
conteo2 = df.groupby(['COLE_DEPTO_UBICACION', 'COLE_BILINGUE']).size().reset_index(name='Count2')
conteo3 = df.groupby(['FAMI_ESTRATOVIVIENDA', 'COLE_BILINGUE']).size().reset_index(name='Count3')
conteo4 = df.groupby('ESTU_GENERO')['PUNT_GLOBAL'].mean().reset_index(name='Count4')
conteo5 = df.groupby('COLE_DEPTO_UBICACION')['PUNT_GLOBAL'].mean().reset_index(name='Count5')


df['PERIODO'] = df['PERIODO'].astype(object)
categorical_columns = df.select_dtypes(include=['object']).columns


PERIODO = df['PERIODO'].unique()
COLE_AREA_UBICACION = df['COLE_AREA_UBICACION'].unique()
COLE_BILINGUE = df['COLE_BILINGUE'].unique()
COLE_CALENDARIO = df['COLE_CALENDARIO'].unique()
COLE_DEPTO_UBICACION = df['COLE_DEPTO_UBICACION'].unique()
COLE_GENERO = df['COLE_GENERO'].unique()
COLE_NATURALEZA = df['COLE_NATURALEZA'].unique()
ESTU_GENERO = df['ESTU_GENERO'].unique()
FAMI_ESTRATOVIVIENDA = df['FAMI_ESTRATOVIVIENDA'].unique()


    
    
# Perform one-hot encoding for each categorical column
for column in categorical_columns:
    one_hot_encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop([column], axis=1)
    
    
X = df.drop('PUNT_GLOBAL', axis=1)

nombresVariables= X.columns
x_prediccion=pd.DataFrame(columns=nombresVariables)
x_prediccion.loc[0] = 0
with open("xgb_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)
    

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Visualizaciones', value='tab-1', children=[
            html.H1(children='Visualizaciones datos icfes', style={'text-align': 'center', 'font-weight': 'bold', 'color': 'blue'}),
            dcc.Graph(
               id='grafico',
               figure=px.bar(conteo, x='COLE_NATURALEZA', y='Count', color='COLE_BILINGUE', barmode='group', title='Comparación de naturaleza del colegio y bilingüismo',
                             labels={'COLE_NATURALEZA': 'Naturaleza del Colegio (Oficial o No Oficial)', 'Count': 'Cantidad de estudiantes'},
                             color_discrete_map={'N': 'blue', 'S': '#7FFFD4'})
            ),
            dcc.Graph(
               id='grafico2',
               figure=px.bar(conteo2, x='COLE_DEPTO_UBICACION', y='Count2', color='COLE_BILINGUE', barmode='group', title='Comparación de departamento colegio y bilingüismo',
                             labels={'COLE_DEPTO_UBICACION': 'Colegios bilingues en departamento ubicación del colegio', 'Count2': 'Cantidad de estudiantes en colegios bilingües'},
                             color_discrete_map={'N': 'blue', 'S': '#7FFFD4'})
            ),
            dcc.Graph(
                id='grafico3',
                figure=px.bar(conteo3, x='FAMI_ESTRATOVIVIENDA', y='Count3', color='COLE_BILINGUE', barmode='group', title='Comparación del estrato y bilingüismo', 
                              labels={'FAMI_ESTRATOVIVIENDA': 'Estratos', 'Count3': 'Cantidad de estudiantes en colegios bilingües'},
                              color_discrete_map={'N': 'blue', 'S': '#7FFFD4'})
            ),
            dcc.Graph(
                id='grafico4',
                figure=go.Figure(data=[
                    go.Bar(x=conteo4['ESTU_GENERO'], y=conteo4['Count4'], marker=dict(color=['#7FFFD4', 'blue']))
                    ],
                    layout=go.Layout(title='Comparación de género y puntaje promedio', xaxis=dict(title='Género'), yaxis=dict(title='Puntaje Promedio'))
                    )
            ),
            dcc.Graph(
                id='grafico5',
                figure=go.Figure(data=[
                    go.Bar(x=conteo5['COLE_DEPTO_UBICACION'], y=conteo5['Count5'], marker=dict(color='lime'))
                    ],
                    layout=go.Layout(
                        title='Comparación de departamento y puntaje promedio',
                        xaxis=dict(title='Departamento Ubicación'),
                        yaxis=dict(title='Puntaje Promedio')
                        )
                    ))
            ]),
        
        dcc.Tab(label='Modelo icfes', value='tab-2', children=[
            html.H1(children='Modelo icfes', style={'text-align': 'center', 'font-weight': 'bold', 'color': 'blue'}),
            html.H1(children=' ', style={ 'background-color': '#7FFFD4', 'padding': '10px'}),
            html.Div([
                html.Label('Periodo', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='PERIODO_',options=PERIODO,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '500px' }),
            html.Div([
                html.Label('Area Ubicación', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_AREA_UBICACION_',options=COLE_AREA_UBICACION,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '50%'}),
            html.Div([
                html.Label('Bilingüe', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_BILINGUE_',options=COLE_BILINGUE,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '500px'}),
            html.Div([
                html.Label('Calendario', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_CALENDARIO_',options=COLE_CALENDARIO,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '50%'}),
            html.Div([
                html.Label('Departamento', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_DEPTO_UBICACION_',options=COLE_DEPTO_UBICACION,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '500px'}),
            html.Div([
                html.Label('Género Colegio', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_GENERO_',options=COLE_GENERO,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '50%'}),
            html.Div([
                html.Label('Naturaleza Colegio', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='COLE_NATURALEZA_',options=COLE_NATURALEZA,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '500px'}),
            html.Div([
                html.Label('Género estudiante', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='ESTU_GENERO_',options=ESTU_GENERO,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '50%'}),
            html.Div([
                html.Label('Estrato estudiante', style={'text-align': 'left', 'font-weight': 'bold', 'color': 'black', 'font-size':'13pt'}),
                dcc.Dropdown(id='FAMI_ESTRATOVIVIENDA_',options=FAMI_ESTRATOVIVIENDA,
                value=None,
                style = {'width': '100px'})
                ], style={'display': 'inline-block', 'width': '500px'}),
            html.H1(children=' ', style={ 'background-color': 'white', 'padding': '10px'}),
            html.H1(children=' ', style={ 'background-color': '#7FFFD4', 'padding': '10px'}),
            html.H1(children=' ', style={ 'background-color': 'white', 'padding': '10px'}),
            
            html.Button('Guardar', id='boton-guardar'),
            html.Div(id='resultado')
            
            
            ])
        ])
    ])
@app.callback(
    dash.dependencies.Output('resultado', 'children'),
    
    [dash.dependencies.Input('boton-guardar', 'n_clicks')],
    [dash.dependencies.State('PERIODO_', 'value'),
     dash.dependencies.State('COLE_AREA_UBICACION_', 'value'),
     dash.dependencies.State('COLE_BILINGUE_', 'value'),
     dash.dependencies.State('COLE_CALENDARIO_', 'value'),
     dash.dependencies.State('COLE_DEPTO_UBICACION_', 'value'),
     dash.dependencies.State('COLE_GENERO_', 'value'),
     dash.dependencies.State('COLE_NATURALEZA_', 'value'),
     dash.dependencies.State('ESTU_GENERO_', 'value'),
     dash.dependencies.State('FAMI_ESTRATOVIVIENDA_', 'value')]
    )
def guardar_datos(n_clicks, periodo, area, bilingue, calendario,depto, generoC,naturaleza, generoE, estrato ):
    if n_clicks is None:
        return ''
    else:
                
        
        for i in nombresVariables:
          if "PERIODO_" + str(periodo) == i:
              x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_AREA_UBICACION_" + area  == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_BILINGUE_" + bilingue == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_CALENDARIO_" + calendario == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_DEPTO_UBICACION_" + depto == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_GENERO_" + generoC == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "COLE_NATURALEZA_" + naturaleza == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "ESTU_GENERO_" + generoE == i:
                x_prediccion[i] = 1
        for i in nombresVariables:
            if "FAMI_ESTRATOVIVIENDA_" + estrato == i:
                x_prediccion[i] = 1
                
        resultado=loaded_model.predict(x_prediccion)

        return f'El puntaje global esperado es de: {resultado}'        

    
              
              
    

    
       
       


if __name__ == '__main__':
    app.run_server(debug=False)
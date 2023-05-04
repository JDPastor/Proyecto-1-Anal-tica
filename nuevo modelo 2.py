import pandas as pd
import networkx as nx
import matplotlib
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

df = pd.read_csv("datosTrain")

print(df.head())
print(df.describe())
print(df.columns)


from pgmpy . estimators import HillClimbSearch
from pgmpy . estimators import K2Score
scoring_method = K2Score ( data =df)
esth = HillClimbSearch ( data =df)
estimated_modelh = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print ( estimated_modelh )
print ( estimated_modelh . nodes () )
print ( estimated_modelh . edges () )

estimated_model = BayesianNetwork(estimated_modelh)
estimated_model.fit(data=df, estimator = MaximumLikelihoodEstimator) 

nx.draw_shell(estimated_model, with_labels=True)

print ( scoring_method . score ( estimated_modelh ) )

from pgmpy . estimators import BicScore

scoring_method = BicScore(data =df)
esth = HillClimbSearch ( data =df)
estimated_modelh = esth.estimate(scoring_method = scoring_method , max_indegree =4 , max_iter =int (1e4))
print ( estimated_modelh )
print ( estimated_modelh . nodes () )
print ( estimated_modelh . edges () )

estimated_model = BayesianNetwork(estimated_modelh)
estimated_model.fit(data=df, estimator = MaximumLikelihoodEstimator) 

nx.draw_shell(estimated_model, with_labels=True)

print ( scoring_method . score ( estimated_modelh ) )

print(estimated_model)

from pgmpy.readwrite import BIFWriter
writer = BIFWriter(estimated_model)
writer.write_bif(filename='modeloNuevoPuntaje.bif')

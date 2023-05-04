
import pandas as pd
import networkx as nx
import matplotlib

df = pd.read_csv("datosTrain")

print(df.head())
print(df.describe())
print(df.columns)

from pgmpy.estimators import PC
est = PC(data=df)

estimated_model = est.estimate(variant="stable", max_cond_vars=4)
print(estimated_model)
print(estimated_model.nodes())
print(estimated_model.edges())

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

estimated_model = BayesianNetwork(estimated_model)
estimated_model.fit(data=df, estimator = MaximumLikelihoodEstimator) 
for i in estimated_model.nodes():
    print(estimated_model.get_cpds(i))

nx.draw_shell(estimated_model, with_labels=True)

from pgmpy.readwrite import BIFWriter
writer = BIFWriter(estimated_model)
writer.write_bif(filename='modeloNuevoRestricciones.bif')


print(estimated_model)

from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination

import pandas as pd
import numpy as np

df = pd.read_csv('datosFinal')
print(df)

model = BayesianNetwork([('thal','trestbps'),('age','trestbps'),('age','chol'),('sex','chol'),('slope','num'),('trestbps','num'),('chol','num'),('fbs','num'),('restecg','num'),('exang','num'),('num','thalach'),('num','cp')])
print(model)


from pgmpy.estimators import MaximumLikelihoodEstimator
emv = MaximumLikelihoodEstimator(model=model, data=df)

# Estimar para nodos sin padres
cpdem_th = emv.estimate_cpd(node="thal")
print(cpdem_th)
cpdem_sl = emv.estimate_cpd(node="slope")
print(cpdem_sl)
cpdem_a = emv.estimate_cpd(node="age")
print(cpdem_a)
cpdem_sx = emv.estimate_cpd(node="sex")
print(cpdem_sx)
cpdem_f = emv.estimate_cpd(node="fbs")
print(cpdem_f)
cpdem_r = emv.estimate_cpd(node="restecg")
print(cpdem_r)
cpdem_ex = emv.estimate_cpd(node="exang")
print(cpdem_ex)

# Estimar para nodos restantes
cpdem_tb = emv.estimate_cpd(node="trestbps")
print(cpdem_tb)
cpdem_c = emv.estimate_cpd(node="chol")
print(cpdem_c)
cpdem_n = emv.estimate_cpd(node="num")
print(cpdem_n)
cpdem_tch = emv.estimate_cpd(node="thalach")
print(cpdem_tch)
cpdem_cp = emv.estimate_cpd(node="cp")
print(cpdem_cp)

model.fit(data=df, estimator = MaximumLikelihoodEstimator) 
for i in model.nodes():
    print(model.get_cpds(i))

from pgmpy.estimators import BayesianEstimator
eby = BayesianEstimator(model=model, data=df)

pseudo_counts = np.ones((2, 64)) * 200000
cpdby_l = eby.estimate_cpd(node="num", prior_type="dirichlet", pseudo_counts=pseudo_counts)
print(cpdby_l)




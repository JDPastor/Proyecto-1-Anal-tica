from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.readwrite import BIFWriter
from pgmpy.readwrite import BIFReader
import networkx as nx

import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('datosTrain')
dfTe = pd.read_csv('datosTest')
labels = dfTe['num']
dfTe = dfTe.drop('num',axis = 1)
print(df)

reader = BIFReader("modeloNuevoPuntaje.BIF")
model = reader.get_model()

model.fit(data=df, estimator = MaximumLikelihoodEstimator) 
for i in model.nodes():
    print(model.get_cpds(i))


graph = nx.DiGraph()

# Add nodes to the graph
graph.add_nodes_from(model.nodes())

# Add edges to the graph
graph.add_edges_from(model.edges())

# Set the layout for visualizing the graph
layout = nx.spring_layout(graph)

# Draw the graph
nx.draw_networkx(graph, pos=layout, with_labels=True, arrows=True)

# Show the graph
plt.show()
print(model.edges)

pred = model.predict(dfTe)
pred = pred['num']

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# compare predicted values with actual values and update counts
for i in range(len(pred)):
    if pred[i] == labels[i]:
        if pred[i] == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        if pred[i] == 1:
            false_positives += 1
        else:
            false_negatives += 1

# calculate percentages
total = true_positives + true_negatives + false_positives + false_negatives
true_positive_rate = true_positives / total
true_negative_rate = true_negatives / total
false_positive_rate = false_positives / total
false_negative_rate = false_negatives / total

print('True positives:', true_positives)
print('True negatives:', true_negatives)
print('False positives:', false_positives)
print('False negatives:', false_negatives)

# compare predicted values with actual values and update counts



pred_prob = pred

accuracy = accuracy_score(labels, pred)
print("Accuracy:", accuracy)


fpr, tpr, thresholds = roc_curve(labels, pred_prob)

# Compute the area under the ROC curve (AUC)
auc = roc_auc_score(labels, pred_prob)
print("AUC:", auc)




# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# Show the legend and plot
plt.legend()
plt.show()


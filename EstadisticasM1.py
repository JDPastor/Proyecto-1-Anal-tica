from pgmpy . models import BayesianNetwork
from pgmpy . factors . discrete import TabularCPD
from pgmpy . inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.readwrite import BIFWriter
from pgmpy.readwrite import BIFReader
import networkx as nx
from sklearn import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import numpy as np

df = pd.read_csv('datosTrain')
datos = pd.read_csv('DatosTest')
dfTe = datos
dfTe = dfTe.to_numpy()
dfTe = dfTe.astype(int)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

inference = VariableElimination(model)

predictions = []
anotacionesK2 = []

for i in range(0, len(dfTe)):
    anotacionesK2 = np.append(anotacionesK2, dfTe[i, 6])
    posterior_pk2 = inference.query(["num"],evidence={"age": dfTe[i,0], "sex": dfTe[i,1], "cp": dfTe[i,2],"trestbps": dfTe[i,3],"chol": dfTe[i,4],"fbs": dfTe[i,5],"restecg": dfTe[i,6],"thalach": dfTe[i,7],"exang": dfTe[i,8],"slope": dfTe[i,9],"thal": dfTe[i,10]})
    probabilidadesk2 = posterior_pk2.values
    
    if np.isnan(probabilidadesk2).any():
        anotacionesK2 = np.delete(anotacionesK2, i, axis=0)

    maximok2 = np.max(probabilidadesk2)

    for j in range(0, len(probabilidadesk2)):
        if probabilidadesk2[j] == maximok2:
            posicion = j
            if posicion == 2:
                predictions = np.append(predictions, 3)
            else:
                predictions = np.append(predictions, posicion)

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# compare predicted values with actual values and update counts
for i in range(len(predictions)):
    if predictions[i] == anotacionesK2[i]:
        if predictions[i] == 1:
            true_positives += 1
        else:
            true_negatives += 1
    else:
        if predictions[i] == 1:
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

pred_prob = predictions

accuracy = accuracy_score(anotacionesK2, predictions)
print("Accuracy:", accuracy)

fpr, tpr, thresholds = roc_curve(anotacionesK2, pred_prob)

# Compute the area under the ROC curve (AUC)
auc = roc_auc_score(anotacionesK2, pred_prob)
print("AUC:", auc)


# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')

# Set anotacionesK2 and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# Show the legend and plot
plt.legend()
plt.show()

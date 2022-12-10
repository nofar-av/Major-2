import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#a
df = pd.read_csv(r'virus_data.csv')

#b
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.2, random_state = (73 + 98)) 

#Visualization and basic analysis - task
DataFrame = train[["PCR_01", "PCR_02", "spread"]] #TODO::NEEDS TO BE NORMALIZED

#Q1
g = sns.jointplot(DataFrame.PCR_01, DataFrame.PCR_02, hue=DataFrame.spread, palette='pastel')
plt.subplots_adjust(top=0.9)
plt.suptitle("PCR_01 and PCR_02 ")
plt.show()

#Q2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
visualize_clf(knn, DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread, "kNN Model", "PCR_01", "PCR_02")


#model selection - Q3
from sklearn.model_selection import cross_validate

k_values  = list(range(1, 20, 2)) + list(range(20, 871, 85))
validation_scores= np.empty(shape=len(k_values), dtype=float)
train_scores = np.empty(shape=len(k_values), dtype=float)

for index, k in enumerate(k_values):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
  results = cross_validate(knn, DataFrame[["PCR_01", "PCR_02"]], y=DataFrame.spread, cv=8,  return_train_score=True)
  train_scores[index] = results["train_score"].mean()
  validation_scores[index] = results["test_score"].mean()

  
print(train_scores)
print(validation_scores)
#plt.semilogx(train_scores, k_values)
plt.semilogx(validation_scores, k_values)
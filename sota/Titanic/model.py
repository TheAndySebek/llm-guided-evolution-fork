# --OPTION--
from catboost import CatBoostClassifier
import numpy as np

class Model:
    def __init__(self):
      self.classifier = CatBoostClassifier(task_type="GPU")
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)

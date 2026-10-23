# --OPTION--
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class Model:
    def __init__(self):
      self.classifier = GradientBoostingClassifier() # might try CatBoost later
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)

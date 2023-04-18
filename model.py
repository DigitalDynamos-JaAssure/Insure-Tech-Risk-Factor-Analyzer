import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

X = df[['WheelBase', 'Length', 'Width', 'Height', 'CurbWeight', 'Price']]
y = df['RiskFactor']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

filename = 'risk_factor_model.pkl'
pickle.dump(rfc, open(filename, 'wb'))

data = np.array([[94.5, 165.7, 64, 51.4, 2221, 9980
                  ]])
print(rfc.predict(data))

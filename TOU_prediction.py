from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

import matplotlib.pyplot as plt

from TOU_analysis import extract_data, preprocessing

def train_model(X_train, y_train):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    filename = 'trained_model.sav'
    pickle.dump(lr, open(filename, 'wb'))

    return filename

file = 'agile_rates_2019.xlsx'
data = extract_data(file)

X_train, X_test, y_train, y_test  = preprocessing(data)
filename = train_model(X_train, y_train)
loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict(X_test)

plt.figure()
plt.scatter(X_test, y_pred, label='Prediction')
plt.scatter(X_test, y_test, label='Actual')
plt.legend()

plt.show()


# result = loaded_model.score(X_test, y_test)
# print(result)

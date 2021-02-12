import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from windowSlider import WindowSlider
from sklearn.model_selection import train_test_split

### Generate the data
N = 600

#generating uneven timestamps
t = np.arange(0, N, 1).reshape(-1,1)
t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
t = np.array(np.round(t, 2)) #rounds the values of the array
# print(t)

#generating the features (predictors)
x1 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x2 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x3 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)

# noise
n = np.round((np.random.random(N) * 2).reshape(-1,1), 2)

# generating the output data (label)
y = np.array([((np.log(np.abs(2 + x1[t])) - x2[t-1]**2) + 0.02*x3[t-3]*np.exp(x1[t-1])) for t in range(len(t))])
y = np.round(y+n, 2)

#Plotting data
# plt.figure()
# plt.plot(t,y)
# plt.show()

# concatenating dataset
dataset = pd.DataFrame(np.concatenate((t, x1, x2, x3, y), axis=1), 
                       columns=['t', 'x1', 'x2', 'x3', 'y'])

# finding the time diff between observations
deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]) for i in range(len(dataset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))

# adding deltaT into the dataset as a feature
dataset.insert(1, '∆t', deltaT)
dataset.head(3)
print(dataset.head(3))

# print(dataset.iloc[:,5])

# split data
# train_set, valid_set, test_set = load_dataset(data_path, dataset, comments=comments)

X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:5], dataset.iloc[:,5], test_size=0.33, shuffle=False)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# # create windows
# w = 5
# train_constructor = WindowSlider()
# train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
#                                                   previous_y=False)

# test_constructor = WindowSlider()
# test_windows = test_constructor.collect_windows(testset.iloc[:,1:],
#                                                 previous_y=False)

# train_constructor_y_inc = WindowSlider()
# train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
#                                                   previous_y=True)

# test_constructor_y_inc = WindowSlider()
# test_windows_y_inc = test_constructor_y_inc.collect_windows(testset.iloc[:,1:],
#                                                 previous_y=True)

# train_windows.head(3)
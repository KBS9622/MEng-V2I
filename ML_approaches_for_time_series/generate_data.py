import numpy as np
import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
from windowSlider import WindowSlider
from sklearn.model_selection import train_test_split
import time

### Generate the data
N = 600
np.random.seed(42)

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
plt.figure()
plt.plot(t,y)
plt.show()

# concatenating dataset
dataset = pd.DataFrame(np.concatenate((t, x1, x2, x3, y), axis=1), 
                       columns=['t', 'x1', 'x2', 'x3', 'y'])

# finding the time diff between observations
deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]) for i in range(len(dataset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))

# adding deltaT into the dataset as a feature
dataset.insert(1, '∆t', deltaT)
dataset.head(3)
print(dataset.head(10))

# print(dataset.iloc[:,5])

# split data
# train_set, valid_set, test_set = load_dataset(data_path, dataset, comments=comments)

# X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:5], dataset.iloc[:,5], test_size=0.33, shuffle=False)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

trainset = dataset.iloc[:402,:]
testset = dataset.iloc[402:,:]
# print(testset)

# create windows
w = 2
train_constructor = WindowSlider(window_size=w)
train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], window_size=w,
                                                  previous_y=False)

test_constructor = WindowSlider(window_size=w)
test_windows = test_constructor.collect_windows(testset.iloc[:,1:],window_size=w,
                                                previous_y=False)


# # window slider with previous y included
# train_constructor_y_inc = WindowSlider()
# train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
#                                                   previous_y=True)

# test_constructor_y_inc = WindowSlider()
# test_windows_y_inc = test_constructor_y_inc.collect_windows(testset.iloc[:,1:],
#                                                 previous_y=True)

print(train_windows.head(3))

# ________________ Y_pred = current Y ________________ 
bl_trainset = cp.deepcopy(trainset)
bl_testset = cp.deepcopy(testset)

bl_y = pd.DataFrame(bl_testset['y'])
bl_y_pred = bl_y.shift(periods=1)

bl_residuals = bl_y_pred - bl_y
bl_rmse = np.sqrt(np.sum(np.power(bl_residuals,2)) / len(bl_residuals))
print('RMSE = %.2f' % bl_rmse)
print('Time to train = 0 seconds')

# ______________ MULTIPLE LINEAR REGRESSION ______________ #
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(trainset.iloc[:,:-1], trainset.iloc[:,-1])

t0 = time.time()
lr_y = testset['y'].values # true y values for test set
lr_y_fit = lr_model.predict(trainset.iloc[:,:-1]) # y of trainset
lr_y_pred = lr_model.predict(testset.iloc[:,:-1]) # predicted y of test set
tF = time.time()

lr_residuals = lr_y_pred - lr_y
lr_rmse = np.sqrt(np.sum(np.power(lr_residuals,2)) / len(lr_residuals))
print('RMSE = %.2f' % lr_rmse)
print('Time to train = %.2f seconds' % (tF - t0))

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,6))
ax1.set_title('Test set')
ax1.plot(testset.iloc[:,0], lr_y, 'b-')
ax1.plot(testset.iloc[:,0], lr_y_pred, 'r-')
ax2.set_title('Train set')
ax2.plot(trainset.iloc[:,0], trainset.iloc[:,-1], 'b-')
ax2.plot(trainset.iloc[:,0], lr_y_fit, 'r-')
plt.show()

# ___________ MULTIPLE LINEAR REGRESSION ON WINDOWS ___________ 
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(train_windows.iloc[:,:-1], train_windows.iloc[:,-1])

t0 = time.time()
lr_y = test_windows['Y'].values
lr_y_fit = lr_model.predict(train_windows.iloc[:,:-1])
lr_y_pred = lr_model.predict(test_windows.iloc[:,:-1])
tF = time.time()

lr_residuals = lr_y_pred - lr_y
lr_rmse = np.sqrt(np.sum(np.power(lr_residuals,2)) / len(lr_residuals))
print('RMSE = %.2f' % lr_rmse)
print('Time to train = %.2f seconds' % (tF - t0))

start_index = train_constructor.w + train_constructor.r - 1
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,6))
ax1.set_title('Test set')
ax1.plot(testset.iloc[start_index:,0], lr_y, 'b-')
ax1.plot(testset.iloc[start_index:,0], lr_y_pred, 'r-')
ax2.set_title('Train set')
ax2.plot(trainset.iloc[start_index:,0], trainset.iloc[start_index:,-1], 'b-')
ax2.plot(trainset.iloc[start_index:,0], lr_y_fit, 'r-')
plt.show()
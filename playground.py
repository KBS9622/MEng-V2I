import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import pandas as pd

# To generate some test data
# x = np.random.randn(500)
# y = np.random.randn(500)

# XY = np.stack((x,y),axis=-1)

# def selection(XY, limitXY=[[-2,+2],[-2,+2]]):
#         XY_select = []
#         for elt in XY:
#             if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
#                 XY_select.append(elt)

#         return np.array(XY_select)

# XY_select = selection(XY, limitXY=[[-2,+2],[-2,+2]])


file_path = 'Device12_formatted.csv'
df = pd.read_csv(file_path)
# df = df.set_index('timestamp')
# start_time = pd.to_datetime('2017-09-25 13:41:41')
# end_time = pd.to_datetime('2017-09-30 23:30:00')
# df = df.loc[0:100000,:]
df = df.loc[200001:300000,:]
print(df['accel_mps2'].max())
x = df['speed_mps'].to_numpy()
y = df['accel_mps2'].to_numpy()

# xAmplitudes = np.array(XY_select)[:,0]#your data here
# yAmplitudes = np.array(XY_select)[:,1]#your other data here


fig = plt.figure() #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')


hist, xedges, yedges = np.histogram2d(x, y, bins=(10,10), range = [[0,+35],[-6,+14]]) # you can change your bins, and the range on which to take data
# hist is a 7X7 matrix, with the populations for each of the subspace parts.
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) -(xedges[1]-xedges[0])


xpos = xpos.flatten()*1./2
ypos = ypos.flatten()*1./2
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
plt.title("speed vs accel for ID 12 200k - 300k Data")
plt.xlabel("speed (mps)")
plt.ylabel("accel (mps2)")
plt.savefig("Device12_200k_to_300k_observations")
plt.show()







# import numpy as np
# import numpy.random
# import matplotlib.pyplot as plt

# # To generate some test data
# x = np.random.randn(500)
# y = np.random.randn(500)

# XY = np.stack((x,y),axis=-1)

# def selection(XY, limitXY=[[-2,+2],[-2,+2]]):
#         XY_select = []
#         for elt in XY:
#             if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
#                 XY_select.append(elt)

#         return np.array(XY_select)

# XY_select = selection(XY, limitXY=[[-2,+2],[-2,+2]])

# heatmap, xedges, yedges = np.histogram2d(XY_select[:,0], XY_select[:,1], bins = 7, range = [[-2,2],[-2,2]])
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


# plt.figure("Histogram")
# #plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin='lower')
# plt.show()
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import pandas as pd

def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"  
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

# file_path = 'Device12_formatted.csv'
# df = pd.read_csv(file_path)
# df = df[(df['speed_mps']!=0) & (df['accel_mps2']!=0)]
# df = df.loc[0:100000,:]
# # df = df.loc[100001:200000,:]
# # df = df.loc[200001:300000,:]
# print(df['speed_mps'].min())
# print(df['speed_mps'].max())
# print(df['accel_mps2'].min())
# print(df['accel_mps2'].max())
# x = df['speed_mps'].to_numpy()
# y = df['accel_mps2'].to_numpy()


file_path = 'test_3d_histo.csv'
df = pd.read_csv(file_path)
df = df[(df['Fake_speed']!=0) & (df['Fake_acc']!=0)]
print(df['Fake_speed'].min())
print(df['Fake_speed'].max())
print(df['Fake_acc'].min())
print(df['Fake_acc'].max())
x = df['Fake_speed'].to_numpy()
y = df['Fake_acc'].to_numpy()

# fig = plt.figure() #create a canvas, tell matplotlib it's 3d
# ax = fig.add_subplot(111, projection='3d')


hist, xedges, yedges = np.histogram2d(x, y, bins=20)#, range = [[0,+30],[-0.5,0.5]]) # you can change your bins, and the range on which to take data
# hist is a 7X7 matrix, with the populations for each of the subspace parts.
print(hist)
print(xedges)
print(yedges)

xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")


# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros_like (xpos)

# dx = xedges [1] - xedges [0]
# dy = yedges [1] - yedges [0]
dz = hist.flatten()

cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# # scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
# plt.title("speed vs accel for ID 12 0 - 100k Data")
# plt.xlabel("speed (mps)")
# plt.ylabel("accel (mps2)")
# plt.savefig("id12_0_to_100k_observations")
# plt.show()
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig,ax=plt.subplots(1,1)
# cp = ax.contourf(xpos, ypos, hist)
# plt.imshow(hist, extent=extent, origin='lower',
#            cmap='RdGy')
# plt.colorbar()
# plt.axis(aspect='image')
# fig.colorbar(cp)
# plt.show()
plt.imshow(hist.T, extent= extent, origin='lower')
plt.colorbar()
plt.show()


# import pickle
# # with open('sinus.pickle', 'wb') as f: # should be 'wb' rather than 'w'
# #     pickle.dump(fig, f) 

# fig_handle = pickle.load(open('sinus.pickle','rb'))
# plt.show(show_figure(fig_handle))



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hist, xedges, yedges = np.histogram2d(x, y, bins=4)
# print(hist)
# print(xedges)
# print(yedges)

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
# print(xpos)
# print(ypos)
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()









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
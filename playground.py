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

# file_path = 'caltrans_processed_drive_cycles/data/1035198_1/2012-05-28.csv'
# df = pd.read_csv(file_path)
# # remove observations when nothing happens as it can drown out the graph
# df = df[(df['speed_mph']!=0) & (df['accel_meters_ps']!=0)]
# df = df.loc[:,['speed_mph','accel_meters_ps']]
# print(df)
# # # convert the units to metres per second from mph
# df = df/2.237
# # df = df.loc[0:100000,:]
# # df = df.loc[100001:200000,:]
# # df = df.loc[200001:300000,:]
# print(df['speed_mph'].min())
# print(df['speed_mph'].max())
# print(df['accel_meters_ps'].min())
# print(df['accel_meters_ps'].max())
# # x axis is the speed
# x = df['speed_mph'].to_numpy()
# # y axis is the acceleration
# y = df['accel_meters_ps'].to_numpy()

# outlier_index = df.loc[df['accel_mps2']>5].index
# print(outlier_index)


file_path = 'device12_oct_7_to_10_classified_updated.csv'
df = pd.read_csv(file_path)

# remove observations when nothing happens as it can drown out the graph
df = df[(df['speed_mps']!=0) & (df['accel_mps2']!=0)]
# convert the units to metres per second from mph
# df = df/2.237
print(df['speed_mps'].min())
print(df['speed_mps'].max())
print(df['accel_mps2'].min())
print(df['accel_mps2'].max())
# x axis is the speed
x = df['speed_mps'].to_numpy()
# y axis is the acceleration
y = df['accel_mps2'].to_numpy()

# # remove observations when nothing happens as it can drown out the graph
# df = df[(df['Real_speed']!=0) & (df['Real_acc']!=0)]
# # convert the units to metres per second from km/h
# df = df/3.6
# print(df['Real_speed'].min())
# print(df['Real_speed'].max())
# print(df['Real_acc'].min())
# print(df['Real_acc'].max())
# # x axis is the speed
# x = df['Real_speed'].to_numpy()
# # y axis is the acceleration
# y = df['Real_acc'].to_numpy()


hist, xedges, yedges = np.histogram2d(x, y, bins=20, range = [[0,+30],[-5,5]]) # you can change your bins, and the range on which to take data
# hist is a 7X7 matrix, with the populations for each of the subspace parts.
# print(hist)
# print(xedges)
# print(yedges)

xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")

# 3D histogram
fig = plt.figure() #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')
xpos = xpos.flatten()
ypos = ypos.flatten()
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
plt.title("speed vs accel for input DCGT Data")
plt.xlabel("speed (mps)")
plt.ylabel("accel (mps2)")
plt.savefig("3D input DCGT 3 day")
plt.show()

# 2D histogram
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
fig,ax=plt.subplots(1,1)
plt.imshow(hist.T, extent= extent, origin='lower')
plt.colorbar()
plt.title("speed vs accel for input DCGT Data")
plt.xlabel("speed (mps)")
plt.ylabel("accel (mps2)")
plt.savefig("2D input DCGT 3 day")
plt.show()

##### Contour
# cp = ax.contourf(xpos, ypos, hist)
# plt.imshow(hist, extent=extent, origin='lower',
#            cmap='RdGy')
# plt.colorbar()
# plt.axis(aspect='image')
# fig.colorbar(cp)
# plt.show()

# import pickle
# # with open('sinus.pickle', 'wb') as f: # should be 'wb' rather than 'w'
# #     pickle.dump(fig, f) 

# fig_handle = pickle.load(open('sinus.pickle','rb'))
# plt.show(show_figure(fig_handle))


# #script to process yun solutions data
# import os
# import pandas as pd
# from datetime import datetime

# class process():

#     def __init__(self, filepath):
#         self.filepath = filepath
    
#     def load_csv(self,filename):
#         path = self.filepath + '/' + filename
#         df = pd.read_csv(path)
#         cols_wanted = ['timeStamp','speed']
#         df = df.loc[:,cols_wanted]

#         return df

# if __name__ == "__main__":
#     filepath = '/Users/koeboonshyang/Desktop/OpenData'
#     process_obj = process(filepath)
#     filename_dec = 'Dec-17/12.0-1.csv'
#     df_dec = process_obj.load_csv(filename_dec)
#     filename_nov = 'Nov-17/12.0-0.csv'
#     df_nov = process_obj.load_csv(filename_nov)
#     filename_nov2 = 'Nov-17/12.0-1.csv'
#     df_nov2 = process_obj.load_csv(filename_nov2)
#     filename_oct = 'Oct-17/12.0.csv'
#     df_oct = process_obj.load_csv(filename_oct)
#     filename_sept = 'Sep-17/12.0.csv'
#     df_sept = process_obj.load_csv(filename_sept)
#     # filename_jan = 'Jan-18/12.0.csv'
#     # df_jan = process_obj.load_csv(filename_jan)

#     df = [df_sept, df_oct, df_nov, df_nov2, df_dec]#, df_jan]
#     df = pd.concat(df, ignore_index = True)
    
#     # @Heejoon, u can start here since u have already combined the csv
#     #removes duplicate rows based on 'timeStamp'
#     df = df.drop_duplicates(subset = 'timeStamp')
#     print(df)
#     df.reset_index(inplace = True)
#     print(df)
#     # df = df[df.index.duplicated()]

#     #identifies the index of the outliers
#     outlier_index = df.loc[df['speed']>200,:].index
#     #loops thrrough the list of outlier index and changes the df so that any outlier is replaced by the previous observation
#     for x in outlier_index:
#         df.loc[x,'speed'] = df.loc[x-1,'speed'] 
        
    
#     df['timestamp'] = pd.to_datetime(df['timeStamp'])

#     # makes new column called timestep to help calculate acceleration
#     df['timestep'] = (df['timestamp'].shift(-1)-df['timestamp']).astype('timedelta64[s]')
#     df['acceleration'] = (df['speed'].shift(-1)-df['speed'])/df['timestep'] #the unit of acceleration is km/h/s, so be cautious when converting

#     #makes the first row observation of acceleration to 0, as there is no previous speed value to calculate acceleration
#     df.loc[-1,'acceleration'] = 0

#     # the speed column is in km/h whereas acceleration column is in km/h/s, need to convert both to m/s and m/s^2 respectively
#     kmph_to_mps = 3.6
#     df['speed_mps'] = df['speed']/kmph_to_mps
#     df['accel_mps2'] = df['acceleration']/kmph_to_mps

#     df = df.set_index('timestamp')
#     print(df)
#     cols_to_drop = ['timeStamp', 'timestep', 'index', 'speed', 'acceleration']
#     df = df.drop(columns=cols_to_drop)
    
#     print(df)

# df.to_csv(r'/Users/koeboonshyang/Desktop/Device12_formatted.csv')


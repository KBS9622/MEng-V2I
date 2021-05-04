import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import datetime

def load_csv_data(file_name, subdir=''):
    """
    Loads data from .csv file in to DataFrame

    :param file_name: .csv file name in string
    :param subdir: optional parameter to specify the subdirectory of the file
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('../')
    print(file_dir)
    for root, dirs, files in os.walk(file_dir):
        if root.endswith(subdir):
            for name in files:
                if name == file_name:
                    file_path = os.path.join(root, name)

    df = pd.read_csv(file_path)

    return df

# # # loads the csv file
# subdir = 'caltrans_processed_drive_cycles/data/1035198_1'
# file_name = '2012-05-22.csv'
# data = load_csv_data(file_name, subdir)

# # get only timestamp and speed_mph for the slice that you want
# data = data.loc[:, ['timestamp','speed_mph']]
# data['timestamp'] = pd.to_datetime(data['timestamp'], dayfirst=True)
# print(data)


# # loads the csv file
subdir = ''
file_name = 'Device12_formatted.csv'
data = load_csv_data(file_name, subdir)

# get only timestamp and speed_mps for the slice that you want
data = data.loc[:, ['timestamp','speed_mps','accel_mps2']]
data['timestamp'] = pd.to_datetime(data['timestamp'], dayfirst=True)
# set the start and end of the slice of data
start_time = pd.to_datetime('09-10-2017 00:00:00', dayfirst=True)
end_time = pd.to_datetime('11-10-2017 00:00:00', dayfirst=True)
data = data[(data['timestamp']>start_time)&(data['timestamp']<end_time)]

print(data)




# minimum gap between observed '0' to determine as a pulse (in seconds)
min_gap = 30
# get only the subset of the data where speed = 0
zero_data = data[data['speed_mps']==0]
print(zero_data)
# reset the index of the df
zero_data.reset_index(inplace=True)
print(zero_data)
print(len(zero_data))

#list to keep track of the start and end indexes of driving pulses
pulse_start = []
pulse_end = []


for i in range(len(zero_data)-1):
    index_diff = zero_data.loc[i+1,'index'] - zero_data.loc[i,'index']
    # if index_diff is more than min_gap, it means that there is driving in between the two index, and we should record it down
    if index_diff > min_gap:
        # calculate the time difference between observed '0's
        time_diff = (zero_data.loc[i+1,'timestamp'] - zero_data.loc[i,'timestamp']).seconds
        # print(time_diff)
        # if the time difference is large enough to suggest that it is a pulse, then record down the start and end index of the pulse
        if time_diff>=min_gap:
            pulse_start.append(zero_data.loc[i,'index'])
            pulse_end.append(zero_data.loc[i+1,'index'])


#create new column to label the start and end of a pulse
# first start with labelling the end of pulse as sometimes, the start time may clash with the end of the previous pulse,
# in which we can just have the start index be placed in the next observation, to keep it all in one column
for idx in pulse_end:
    print(idx)
    data.loc[idx,'points'] = 4
for idx in pulse_start:
    print(idx)
    if data.loc[idx,'points'] == 4:
        data.loc[idx+1,'points'] = 1
    else:
        data.loc[idx,'points'] = 1


# list to keep track of all the indexes for the pulse
all_cruise_time = []
cruise_time = []
flag = 0

def onclick(event):
    global ix, iy
    # keep track of the timestamp where the mouse clicked
    ix, iy = matplotlib.dates.num2date(event.xdata).strftime('%Y-%m-%d %H:%M:%S'), event.ydata
    print ('x = {}, y = {}'.format(ix, iy))

    global cruise_time
    # keep track of the datapoints which was clicked in the figure
    cruise_time.append(ix)

    # when the two points have been identified/clicked
    if len(cruise_time) == 2:
        fig.canvas.mpl_disconnect(cid)
        # append the index points to the main list
        all_cruise_time.append(cruise_time)
        # clear the figure's list of two points
        cruise_time = []
        global flag
        # flag to indicate that the loop can proceed
        flag = 1
        # close the figure
        plt.close()

    return cruise_time

print(len(pulse_start))
print(len(pulse_end))

# loop over the number of pulses in the df
for i in range(len(pulse_start)):
    # get the slice for the driving pulse
    x = data.loc[pulse_start[i]:pulse_end[i],'timestamp']
    y = data.loc[pulse_start[i]:pulse_end[i],'speed_mps']
    
    while flag==0:
        # plot the figure
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    # clear the flag
    flag = 0

print(all_cruise_time)

# fill the 'points' column to label the start and end of cruising phase
# the earlier of the two timestamps is the start and the later is the end of the cruising phase
for idx in all_cruise_time:
    cruise_start = min(idx)
    cruise_end = max(idx)
    # get the index for the start and end of the cruising phase
    cruise_start_idx = data[data['timestamp']==cruise_start].index
    cruise_end_idx = data[data['timestamp']==cruise_end].index
    # note it in the dataframe 
    data.loc[cruise_start_idx,'points'] = 2
    data.loc[cruise_end_idx,'points'] = 3

data.to_csv(r'/Users/koeboonshyang/Desktop/device12_oct_classifed.csv', index = False)
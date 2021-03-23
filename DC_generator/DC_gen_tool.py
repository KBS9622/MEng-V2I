import os
import math
from sympy import *
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.optimize import minimize as sp_minimize
from scipy import special
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
import lmfit
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

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

def Gaussian_param():
    """ Inital parameters for each Gaussian characteristic paramter 
    """
    LMparams = Parameters()
    # there are two sets of values because the paper used 2 curve (n=2) to fit the distributions
    LMparams.add('alpha_1', value = 1., min = 0)
    LMparams.add('alpha_2', value = 1., min = 0)
    LMparams.add('sigma_1', value = 1., min = 0)
    LMparams.add('sigma_2', value = 1., min = 0)
    LMparams.add('meu_1', value = 1., min = 0)
    LMparams.add('meu_2', value = 1., min = 0)

    return LMparams

def Gaussian_idle_param():
    """ Inital parameters for each Gaussian characteristic paramter 
    """
    LMparams = Parameters()
    # there are two sets of values because the paper used 2 curve (n=2) to fit the distributions
    LMparams.add('alpha_1', value = 100., min = 0)
    LMparams.add('alpha_2', value = 50., min = 0)
    LMparams.add('sigma_1', value = 100., min = 0)
    LMparams.add('sigma_2', value = 100., min = 0)
    LMparams.add('meu_1', value = 2500., min = 0)
    LMparams.add('meu_2', value = 5000., min = 0)

    return LMparams

def LF_Noise(component = 1):
    """ Inital parameters and bounds for each paramter according to LF (0-0.01Hz)in the paper
    """
    LMparams = Parameters()

    # The code below is to load the initial paramters if we are running the NLLSR on all 3 components at once

    LMparams.add('A1_FS', value = 10.)
    LMparams.add('A2_FS', value = 10.)
    LMparams.add('A3_FS', value = 10.)
    LMparams.add('w1_FS', value = 0, min = 0, max = 0.01*2*math.pi)
    LMparams.add('w2_FS', value = 0.005*2*math.pi, min = 0, max = 0.01*2*math.pi)
    LMparams.add('w3_FS', value = 0.01*2*math.pi, min = 0, max = 0.01*2*math.pi)
    LMparams.add('phi1_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi2_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi3_FS', value = 0, min = -math.pi, max = math.pi)

    # The code below is to load the initial paramters if we are running the NLLSR on one component at a time

    # if component == 1:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0, min = 0, max = 0.01*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 2:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0.005*2*math.pi, min = 0, max = 0.01*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 3:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0.01*2*math.pi, min = 0, max = 0.01*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)

    return LMparams

def MF_Noise(component = 1):
    """ Inital parameters and bounds for each paramter according to MF (0.01-0.25Hz) in the paper
    """
    LMparams = Parameters()

    # The code below is to load the initial paramters if we are running the NLLSR on all 3 components at once

    LMparams.add('A1_FS', value = 10.)
    LMparams.add('A2_FS', value = 10.)
    LMparams.add('A3_FS', value = 10.)
    LMparams.add('w1_FS', value = 0.02*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    LMparams.add('w2_FS', value = 0.03*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    LMparams.add('w3_FS', value = 0.03*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    LMparams.add('phi1_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi2_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi3_FS', value = 0, min = -math.pi, max = math.pi)

    # The code below is to load the initial paramters if we are running the NLLSR on one component at a time

    # if component == 1:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0.02*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 2:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0.03*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 3:
    #     LMparams.add('A_FS', value = 10.)
    #     LMparams.add('w_FS', value = 0.03*2*math.pi, min = 0.01*2*math.pi, max = 0.25*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)

    return LMparams

def HF_Noise(component = 1):
    """ Inital parameters and bounds for each paramter according to HF (0.25-0.5Hz) in the paper
    """
    LMparams = Parameters()

    # The code below is to load the initial paramters if we are running the NLLSR on all 3 components at once

    LMparams.add('A1_FS', value = 1.)
    LMparams.add('A2_FS', value = 1.)
    LMparams.add('A3_FS', value = 1.)
    LMparams.add('w1_FS', value = 0.25*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    LMparams.add('w2_FS', value = 0.375*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    LMparams.add('w3_FS', value = 0.5*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    LMparams.add('phi1_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi2_FS', value = 0, min = -math.pi, max = math.pi)
    LMparams.add('phi3_FS', value = 0, min = -math.pi, max = math.pi)

    # The code below is to load the initial paramters if we are running the NLLSR on one component at a time

    # if component == 1:
    #     LMparams.add('A_FS', value = 1.)
    #     LMparams.add('w_FS', value = 0.25*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 2:
    #     LMparams.add('A_FS', value = 1.)
    #     LMparams.add('w_FS', value = 0.375*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)
    # elif component == 3:
    #     LMparams.add('A_FS', value = 1.)
    #     LMparams.add('w_FS', value = 0.5*2*math.pi, min = 0.25*2*math.pi, max = 0.5*2*math.pi)
    #     LMparams.add('phi_FS', value = 0, min = -math.pi, max = math.pi)

    return LMparams


class DP(object):
    """ Module 3: Drive Pulse
    """

    # def __init__(self, pulse_duration):
    #     """ Constructor takes in pulse duration and creates instances of inverse cdfs for each of the intersted parameters needed to
    #         generate the drive pulse. Creates self.parameter dictionary containing all relevant parameters.
    #     """
    #     self.pulse_duration = pulse_duration 
    #     self.total_t = np.linspace(0, 1000*self.pulse_duration, (1000*self.pulse_duration)+1)
    #     self.accel_inv_cdf_obj = self.create_inv_cdf_objects(Acceleration())
    #     self.cruising_duration_inv_cdf_obj = self.create_inv_cdf_objects(Cruising_Duration())
    #     self.avg_cruising_speed_inv_cdf_obj = self.create_inv_cdf_objects(Average_Crusing_Speed())
    #     self.decel_inv_cdf_obj = self.create_inv_cdf_objects(Deceleration())
    #     self.random_select_params()
    #     self.params = self.parameters_for_drive_cycle(self.accel_value, self.decel_value, self.cruising_duration_value, self.avg_cruising_speed_value)
    #     print(self.params)

    def __init__(self, pulse_duration, extract_obj, seed=10):
        """ Constructor takes in pulse duration and extraction obj, then creates instances of inverse cdfs for each of the interested
        parameters needed to generate the drive pulse. Creates self.parameter dictionary containing all relevant parameters.
        """
        np.random.seed(seed)
        self.pulse_duration = pulse_duration
        self.total_t = np.linspace(0, 1000 * self.pulse_duration, (1000 * self.pulse_duration) + 1)
        self.accel_inv_cdf_obj = self.create_inv_cdf_objects(extract_obj.accel_obj)
        self.cruising_duration_inv_cdf_obj = self.create_inv_cdf_objects(extract_obj.cd_obj)
        self.avg_cruising_speed_inv_cdf_obj = self.create_inv_cdf_objects(extract_obj.avg_cs_obj)
        self.decel_inv_cdf_obj = self.create_inv_cdf_objects(extract_obj.decel_obj)

    def create_inv_cdf_objects(self, attribute_obj, num_of_gauss=2):
        """
        Takes in one of parameter's classes needed for driving pulse and instantiates object for the inverse_cdf
        """
        # load initial gaussian parameters
        param_obj = Gaussian_param()
        # create a probability function object for attribute with its attribute histogram data
        attribute_prob_obj = Probability_Functions(attribute_obj.bins, attribute_obj.data_points, num_of_gauss)
        # fit the histogram
        fitted_obj = attribute_prob_obj.NLLSR(param_obj)
        # create inverse cdf object
        inv_cdf_obj = inv_cdf(attribute_prob_obj)

        return inv_cdf_obj

    def random_select_params(self):
        """
        Randomly generates numbers from 0 to 1 to randomly select values for parameters using their inverse cdf
        """
        list1 = np.random.rand(4).tolist()
        random_numbers = [round(element, 2) for element in list1]
        print(random_numbers)
        self.accel_value = self.accel_inv_cdf_obj.get_value(random_numbers.pop())[0]
        self.cruising_duration_value = self.cruising_duration_inv_cdf_obj.get_value(random_numbers.pop())[0]
        # self.cruising_duration_value = self.cruising_duration_inv_cdf_obj.get_value(0.98)[0]
        self.avg_cruising_speed_value = self.avg_cruising_speed_inv_cdf_obj.get_value(random_numbers.pop())[0]
        self.decel_value = self.decel_inv_cdf_obj.get_value(random_numbers.pop())[0]

    def crusing_with_noise(self, time_array, velocity_noise_obj):
        """
        time_array: array of timestamps to compute for corresponding velocity noise using fitted model
        velocity_noise_obj: VN object containing fitted model and respective parameters for 3 freq components

        return:: cruising_with_noise set of speed values containing average cruising speed superimposed on velocity noise
        """
        # resets time to match the specified duration of the driving pulse
        velocity_noise_obj.set_t(time_array)
        # returns velocity noise speed values (y axis)
        velocity_noise = velocity_noise_obj.final_curve()
        # scale the velocity noise so that velocity during crusing does not go negative
        scaled_velocity_noise = self.scale_velocity_noise(velocity_noise)
        # adds velocity noise speed values (y axis) to static cruising speed
        cruising_with_noise = scaled_velocity_noise + self.params["cruising speed"]

        plt.plot(time_array, cruising_with_noise)
        plt.show()
        return cruising_with_noise

    def scale_velocity_noise(self, velocity_noise):
        # get the minimum velocity of the velocity noise component only (centered around '0')
        min_vn = min(velocity_noise)
        # get the minimum velocity of velocity noise with cruising speed
        min_cruise_with_vn = min_vn + self.params["cruising speed"]
        # only scale the velocity noise if it results in a negative cruising with vn speed value
        if min_cruise_with_vn <= 0:
            # find the scale ratio that causes the lowest of the cruise with vn speed value to be 0
            scale = self.params["cruising speed"] / (-min_vn)
            print(scale)
            # round the value down
            scale = int(math.floor(100 * scale))
            print(scale)
            # scale the vn
            scaled_vn = (scale / 100) * velocity_noise
            return scaled_vn
        return velocity_noise

    def parameters_for_drive_cycle(self, acceleration, decceleration, cruising_duration, average_cruising_speed):
        """
        accepts as parameters the 4 randomly selected values from the inverse cdfs, computes other parameters and returns
        this as a dictionary
        """
        # computes acceleration time by using average cruising speed as initial speed of cruising duration
        acceleration_time = 1000 * average_cruising_speed / acceleration
        # computes decceleration time by using average cruising speed as final speed of cruising duration
        decceleration_time = 1000 * average_cruising_speed / decceleration
        # computes idle time by subtracting all other durations from total pulse duration
        idle_time = (1000 * self.pulse_duration) - acceleration_time - decceleration_time - (1000 * cruising_duration)

        # constructs parameters dictionary containing everything needed to generate driving pulse
        parameters = {"acceleration": acceleration,
                      "decceleration": decceleration,
                      "acceleration duration": round(acceleration_time),
                      "decceleration duration": round(decceleration_time),
                      "cruising duration": round(1000 * cruising_duration),
                      "cruising speed": average_cruising_speed,
                      "idle duration": idle_time,
                      "total duration": 1000 * self.pulse_duration
                      }
        return parameters

    def generate_driving_pulse(self, velocity_noise_obj):
        """ Function to be called from outside the class that outputs plot of generated driving pulse
        """
        self.random_select_params()
        self.params = self.parameters_for_drive_cycle(self.accel_value, self.decel_value, self.cruising_duration_value,
                                                      self.avg_cruising_speed_value)
        print(self.params)
        # Call cruising_with_noise method to return corresponding values (y axis) for cruising
        # Inputs the whole pulse duration as cruise duration therefore extra values are present
        # print(self.total_t)
        # print(self.params["acceleration duration"])
        # print(np.where(self.total_t[:]==self.params["acceleration duration"]))
        # print(np.where(self.total_t[:]==self.params["acceleration duration"])[0][0])
        # print(self.total_t[np.where(self.total_t[:]==self.params["acceleration duration"])[0][0]:]/1000)
        speed_while_cruising_extra_values = self.crusing_with_noise(self.total_t[np.where(
            self.total_t[:] == self.params["acceleration duration"])[0][0]:np.where(
            self.total_t[:] == (self.params["acceleration duration"] + self.params["cruising duration"]))[0][0]] / 1000,
                                                                    velocity_noise_obj)
        # computes initial cruising speed with velocity noise
        initial_cruising_speed = speed_while_cruising_extra_values[0]
        # print('initial_cruising_speed:{}'.format(initial_cruising_speed))
        # print(self.params["acceleration"])
        # computes actual acceleartion duration using the caluclated initial speed
        # no longer is using the estimate of initial= avergae cruising speed as in the parameters_for_drive_cycle method
        self.params["acceleration duration"] = round(1000 * initial_cruising_speed / self.params["acceleration"])
        # print('accel_duration:{}'.format(self.params["acceleration duration"]))
        # print(self.total_t)
        # print(1000*self.params["acceleration duration"])
        # print(np.where(self.total_t[:]==self.params["acceleration duration"]))

        # Retrieves x axis (time steps) and y axis (speed) values during acceleration period
        accel_time_values = self.total_t[:np.where(self.total_t[:] == self.params["acceleration duration"])[0][0]]
        speed_during_acceleration = self.params["acceleration"] * accel_time_values / 1000
        current_time = self.params["acceleration duration"]

        # Retrieves x axis (time steps) and y axis (speed) values during cruising period
        cruising_time_values = self.total_t[np.where(self.total_t[:] == current_time)[0][0]:np.where(
            self.total_t[:] == current_time + self.params["cruising duration"])[0][0]]
        speed_during_cruising = speed_while_cruising_extra_values[
                                :np.where(self.total_t[:] == self.params["cruising duration"])[0][0]]
        current_time += self.params["cruising duration"]

        # Retrieves x axis (time steps) and y axis (speed) values during decceleration period
        final_cruising_speed = speed_during_cruising[-1]
        # print(final_cruising_speed)
        # print(self.params["decceleration"])
        self.params["decceleration duration"] = round(1000 * final_cruising_speed / self.params["decceleration"])
        # print(self.params["decceleration duration"])
        end_time = round(current_time + self.params["decceleration duration"])
        deccel_time_values = self.total_t[
                             np.where(self.total_t[:] == current_time)[0][0]:np.where(self.total_t[:] == end_time)[0][
                                 0]]
        speed_during_decceleration = final_cruising_speed - (
                    self.params["decceleration"] / 1000 * np.linspace(1, self.params["decceleration duration"],
                                                                      int(self.params["decceleration duration"])))
        current_time += self.params["decceleration duration"]
        # current_time = round(current_time,3)

        # Retrieves x axis (time steps) and y axis (speed= 0) values during idle_time period
        idle_time_values = self.total_t[np.where(self.total_t[:] == current_time)[0][0]:
                                        np.where(self.total_t[:] == self.pulse_duration)[0][0]]
        idle_time = 0 * idle_time_values

        # Plot all 4 periods on same plot to visuale driving pulse
        plt.plot(accel_time_values, speed_during_acceleration, 'r')
        plt.plot(cruising_time_values, speed_during_cruising, 'b')
        plt.plot(deccel_time_values, speed_during_decceleration, 'g')
        plt.plot(idle_time_values, idle_time, 'r')
        plt.show()

        time_steps = np.concatenate((accel_time_values, cruising_time_values, deccel_time_values, idle_time_values))
        speed = np.concatenate(
            (speed_during_acceleration, speed_during_cruising, speed_during_decceleration, idle_time))

        df = pd.DataFrame({'time_steps': time_steps, 'speed': speed})

        df.to_csv('generated_data.csv', index=False, header=True)


class Attribute(object):
    """ Generates a histogram according to the input array
    """
    def __init__(self, array):
        """ Takes in a numpy array and generates a histogram then repeat the datapoints to get a smooth and consistent graph
        Zero frequency bins are added to both side of the histogram to help Gaussian fitting
        """
        self.array = array
        # generates a histogram from the input numpy array
        hist, bin_edges = np.histogram(array)
        # get the bin width of the histogram
        bin_width = bin_edges[1]-bin_edges[0]
        # calculate the number of bins in the histogram
        num_of_bins = len(bin_edges)-1
        # how granular do we want the datapoints to be
        repeat_factor = 100
        # zero bins on both side of histogram to help gaussian fit better
        num_of_zero_bins = 4 # on both sides total
        # new total number of bins in the histogram including zero bins
        new_num_of_bins = num_of_bins + num_of_zero_bins
        # zero frequency arrays for the zero bins
        zero_array = np.repeat([0],num_of_zero_bins/2)
        # concatenate zero arrays and hist to form new hist
        new_hist = np.concatenate((zero_array,hist,zero_array))
        # new first bin edge
        new_first_bin_edge = bin_edges[0]-((num_of_zero_bins/2)*bin_width)
        # new last bin edge
        new_last_bin_edge = bin_edges[-1]+((num_of_zero_bins/2)*bin_width)
        # generate fine 'bins' based on inital bin numbers and repeat factor
        self.bins = pd.DataFrame(np.linspace(new_first_bin_edge, new_last_bin_edge-(bin_width/repeat_factor), num=repeat_factor*new_num_of_bins))
        # repeat the 'frequency' values by the repeat factor
        self.data_points = pd.DataFrame(np.repeat(new_hist,repeat_factor))

class Extract_Hist(object):
    """ This object handles all the extraction of information from labelled data and generates histogram objects based on the data
    """
    def __init__(self, file_name, subdir=''):
        """ Constructor takes in the file name and subdirectory to locate the file and save it as a pandas df
        Then subsets the data by labelled data and completely labelled data and extract relevant attribute information to form histograms
        """
        self.df = load_csv_data(file_name=file_name,subdir=subdir)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], dayfirst=True)
        # identify the labelled points in the data
        self.identify_points()
        # extract all the required attribute information and save the histogram as a local variable
        self.extract_acceleration()
        self.extract_cruise_duration()
        self.extract_avg_cruise_speed()
        self.extract_deceleration()
        self.extract_idle_duration()
    
    def identify_points(self):
        """ Indentifies and subsets the labelled observations in the data and then subset to a smaller df with completely labelled pulses
        Completely labelled pulses are pulses where the label points are in the order '1,2,3,4' and is not missing any values in between
        """
        self.points = self.df.copy()
        # only get the observations for which the points are labeled
        self.points = self.points[self.points['points']>0]
        # reset the index
        self.points.reset_index(inplace=True)
        #print out the values of points to visually determine if there are any errors in labelling order
        # print(self.points['points'].to_numpy())
        # create a list to store only completely labelled pulses
        self.complete_labels = pd.DataFrame(columns=self.points.columns)
        # print(self.complete_labels)
        # identify which observations are not fully labeled (1 and 4 labelled but not 2 and 3)
        i = 0
        while True:
            if i+3 > len(self.points):
                # Condition to break out of the while loop when entire df is gone through, or the last pulse is incomplete
                break
            # create dummy variable thats a slice of the whole data
            temp = self.points.loc[i:i+3]
            if temp.loc[i,'points'] == 1:
                if temp.loc[i+1,'points'] == 2:
                    if temp.loc[i+2,'points'] == 3:
                        if temp.loc[i+3,'points'] == 4:
                            # this means the entire slice is correctly ordered and nothing is missing, 
                            # therefore append to the complete_labels df
                            self.complete_labels = self.complete_labels.append(temp, ignore_index=True)
                            # shift index by four for the next 4 values
                            i += 4
                            # skip to the next iteration of loop
                            continue
            # skip the first observation of the temp df because the four consecutive values do not line up
            i += 1
            # prints incomplete when there is an incompletely labelled pulse
            print('incomplete')

    def extract_acceleration(self):
        """ Extracts acceleration data and generates a histogram object to store the data
        """
        # get the timestamp and speed for points = 1 (start of acceleration phase)
        accel_start = self.complete_labels[self.complete_labels['points']==1]
        accel_start = accel_start.loc[:,['timestamp','speed_mps']]
        # reset the index
        accel_start.reset_index(drop = True, inplace=True)
        # get the timestamp and speed for points = 2 (end of acceleration phase)
        accel_end = self.complete_labels[self.complete_labels['points']==2]
        accel_end = accel_end.loc[:,['timestamp','speed_mps']]
        # reset the index
        accel_end.reset_index(drop = True, inplace=True)
        
        # empty list to store acceleration values
        accel_values = []
        for i in range(len(accel_start)):
            # calculate time difference
            time_diff = (accel_end.loc[i,'timestamp'] - accel_start.loc[i,'timestamp']).seconds
            # calculate speed difference
            speed_diff = accel_end.loc[i,'speed_mps'] - accel_start.loc[i,'speed_mps']
            # calculate acceleration value
            accel = speed_diff/time_diff
            # append the values
            accel_values.append(accel)
        # generates and saves the acceleration histogram object
        self.accel_obj = Attribute(np.array(accel_values))

        return self.accel_obj

    def extract_deceleration(self):
        """ Extracts deceleration data and generates a histogram object to store the data
        """
        # get the timestamp and speed for points = 3 (start of deceleration phase)
        decel_start = self.complete_labels[self.complete_labels['points']==3]
        decel_start = decel_start.loc[:,['timestamp','speed_mps']]
        # reset the index
        decel_start.reset_index(drop = True, inplace=True)
        # get the timestamp and speed for points = 4 (end of deceleration phase)
        decel_end = self.complete_labels[self.complete_labels['points']==4]
        decel_end = decel_end.loc[:,['timestamp','speed_mps']]
        # reset the index
        decel_end.reset_index(drop = True, inplace=True)
        
        # empty list to store deceleration values
        decel_values = []
        for i in range(len(decel_start)):
            # calculate time difference
            time_diff = (decel_end.loc[i,'timestamp'] - decel_start.loc[i,'timestamp']).seconds
            # calculate speed difference
            speed_diff = decel_end.loc[i,'speed_mps'] - decel_start.loc[i,'speed_mps']
            # calculate deceleration value (needs to be positive for rest of code)
            decel = -(speed_diff/time_diff)
            # append the values
            decel_values.append(decel)
        # generates and saves the deceleration histogram object
        self.decel_obj = Attribute(np.array(decel_values))

        return self.decel_obj

    def extract_cruise_duration(self):
        """ Extracts cruising duration data and generates a histogram object to store the data
        """
        # get the timestamp for points = 2 (start of cruise phase)
        cruise_start = self.complete_labels[self.complete_labels['points']==2]
        cruise_start = cruise_start.loc[:,['timestamp']]
        # reset the index
        cruise_start.reset_index(drop = True, inplace=True)
        # get the timestamp for points = 3 (end of cruise phase)
        cruise_end = self.complete_labels[self.complete_labels['points']==3]
        cruise_end = cruise_end.loc[:,['timestamp']]
        # reset the index
        cruise_end.reset_index(drop = True, inplace=True)

        # empty list to store cruising duration values
        cruise_duration_values = []
        for i in range(len(cruise_start)):
            # calculate time difference
            cruise_duration = (cruise_end.loc[i,'timestamp'] - cruise_start.loc[i,'timestamp']).seconds
            # append the values
            cruise_duration_values.append(cruise_duration)
        # generates and saves the crusing duration histogram object
        self.cd_obj = Attribute(np.array(cruise_duration_values))

        return self.cd_obj

    def extract_avg_cruise_speed(self):
        """ Extracts average crusing speed data and generates a histogram object to store the data
        Extracts the cruising phase with velocity noise data and saves it
        """
        # get the timestamp and speed for points = 2 (start of cruise phase)
        cruise_start = self.complete_labels[self.complete_labels['points']==2]
        cruise_start = cruise_start.loc[:,['index']]
        # reset the index
        cruise_start.reset_index(drop = True, inplace=True)
        # get the timestamp and speed for points = 3 (end of cruise phase)
        cruise_end = self.complete_labels[self.complete_labels['points']==3]
        cruise_end = cruise_end.loc[:,['index']]
        # reset the index
        cruise_end.reset_index(drop = True, inplace=True)

        # empty list to store crusing phase with velocity noise data
        self.cruise_with_vn = []
        # empty list to store average cruising speed values
        avg_cruise_speed_values = []
        for i in range(len(cruise_start)):
            # get the original dataset index for start and end of cruise phase
            start_index = cruise_start.loc[i,['index']].values[0]
            end_index = cruise_end.loc[i,['index']].values[0]
            # saves cruising phase with noise as a list of pandas series
            self.cruise_with_vn.append(self.df.loc[start_index:end_index,'speed_mps'])
            # calculate average cruise speed over the cruising phase
            avg_cruise_speed = self.cruise_with_vn[i].mean()
            # append the values
            avg_cruise_speed_values.append(avg_cruise_speed)
        # generates and saves the average crusing speed histogram object
        self.avg_cs_obj = Attribute(np.array(avg_cruise_speed_values))

        return self.avg_cs_obj

    def extract_idle_duration(self):
        """ Extracts idle duration data and generates a histogram object to store the data
        """
        # get the timestamp for points = 4 (start of idle phase)
        idle_start = self.points[self.points['points']==4]
        idle_start = idle_start.loc[:,['timestamp']]
        # the last labelled '4' is the end of a pulse with no subsequent pulse, therefore cannot be used to calculate idle
        idle_start = idle_start.iloc[:-1,:]
        # reset the index
        idle_start.reset_index(drop = True, inplace=True)
        # get the timestamp for points = 1 (end of idle phase)
        idle_end = self.points[self.points['points']==1]
        idle_end = idle_end.loc[:,['timestamp']]
        # the first labelled '1' is the start of a pulse with no prior pulse, therefore cannot be used to calculate idle
        idle_end = idle_end.iloc[1:,:]
        # reset the index
        idle_end.reset_index(drop = True, inplace=True)

        # empty list to store cruising duration values
        idle_duration_values = []
        for i in range(len(idle_start)):
            # calculate time difference
            idle_duration = (idle_end.loc[i,'timestamp'] - idle_start.loc[i,'timestamp']).seconds
            # append the values
            idle_duration_values.append(idle_duration)
        # generates and saves the idle duration histogram object
        self.idle_obj = Attribute(np.array(idle_duration_values))

        return self.idle_obj

class Probability_Functions(object):
    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.original_y = y
        # the number of Gaussian distribution used to describe the data
        self.n = n

    def single_component(self, alpha_i, sigma_i, meu_i):
        """ Returns the single Gaussian component as described in the sum of eqn (10)
        """
        exp_component = -((self.x - meu_i)**2)/(2*(sigma_i**2))
        fcn = (alpha_i/(sigma_i*math.sqrt(2*math.pi)))*np.exp(exp_component)
        return fcn

    def eqn_model(self, params):
        """ Returns the Gaussian component sum as described in eqn (10)
        """
        # runs a loop for the number of of Gaussian distibutions used to describe the data, and sums the single components 
        for component in range(1,self.n+1):
            alpha_i = 'alpha_'+str(component)
            sigma_i = 'sigma_'+str(component)
            meu_i = 'meu_'+str(component)
            if component == 1:
                model = self.single_component(params[alpha_i],params[sigma_i],params[meu_i])
            else:
                model += self.single_component(params[alpha_i],params[sigma_i],params[meu_i])

        return model

    def fnc2min(self, params):
        """ Returns the residuals for eqn_model
        """
        return (self.y - self.eqn_model(params))

    def NLLSR(self, LMparams):
        """ Returns the result of the NLLSR using LMFit
        """
        # uses least swuares method to minimize the parameters given by LMparams according to the residuals given by self.fnc2min
        LMFitmin = Minimizer(self.fnc2min, LMparams)
        LMFitResult = LMFitmin.minimize(method='least_squares')
        lmfit.printfuncs.report_fit(LMFitResult.params)
        self.params = LMFitResult.params

        return LMFitResult

    def single_cdf_component(self, x, alpha_i, sigma_i, meu_i):
        """ Returns the single cdf component as described in the sum of eqn (12)
        """
        erf_param = (x - meu_i)/(sigma_i * math.sqrt(2))
        answer =  0.5 * alpha_i * (1 + special.erf(erf_param))

        return answer

    def cdf(self, x, params):
        """ Returns the cdf component sum as described in eqn (12)
        """
        for component in range(1,self.n+1):
            alpha_i = 'alpha_'+str(component)
            sigma_i = 'sigma_'+str(component)
            meu_i ='meu_'+str(component)

            if component == 1:
                answer =  self.single_cdf_component(x, params[alpha_i], params[sigma_i], params[meu_i])
            else:
                answer +=  self.single_cdf_component(x, params[alpha_i], params[sigma_i], params[meu_i])

        return answer

    def normalised_cdf(self,x, LMparams):
        """ Returns a normalised cdf as described in eqn (13)
        """
        lim_inf_cdf = self.cdf(math.inf, LMparams)
        answer = self.cdf(x, LMparams)/lim_inf_cdf

        return answer

    def normalised_single_cdf(self,x, alpha_i, sigma_i, meu_i):
        """ normalised cdf for a SINGLE component 
        """
        lim_inf_cdf = self.single_cdf_component(math.inf, alpha_i, sigma_i, meu_i)
        answer = self.single_cdf_component(x, alpha_i, sigma_i, meu_i)/lim_inf_cdf

        return answer

    def single_quantile_component(self, p, alpha_i, sigma_i, meu_i):
        """ single inverse cdf component
        """
        inv_erf_param = (2*p/alpha_i) - 1
        answer =  meu_i + (sigma_i * math.sqrt(2) * special.erfinv(inv_erf_param))

        return answer

class inv_cdf(object):
    def __init__(self, prob_obj):
        # create a numpy array ranging from 0 to 1 with granularity of e-3
        # the array represents the possible p values that could be input to determine the attribute value, therefore
        # granularity of the random number generator needs to be set to e-3
        self.x = np.arange(0,1.001,0.001)
        # genenrate an empty array with same shape as self.x
        self.y = np.zeros(self.x.shape)
        # save a local version of the probability object
        self.prob_obj = prob_obj
        # generate the lookup table for the ranges of x
        self.fit()

    def diff(self, x, a):
        """ residual(?) for the fit method
        """
        yt = self.prob_obj.normalised_cdf(x, self.prob_obj.params)
        return (yt - a )**2

    def fit(self):
        """ Fits the inverse normalised cdf and generates a lookup table of sort
        """
        # the for loop generates a lookup table correlating x and y based on the normalised cdf
        for idx,x_value in enumerate(self.x):
            res = sp_minimize(self.diff, 1.0, args=(x_value), method='Nelder-Mead', tol=1e-6)
            self.y[idx] = res.x[0]

    def get_value(self, p):
        """ Gives corresponding attribute value depending on p (ranging from 0 to 1)
        """
        x_copy = np.copy(self.x)
        # find the index at which the x is equal to the input p
        index = np.where(x_copy == p)
        # gets the corresponding attribute value depending on the index
        value = self.y[index]

        return value

class Velocity_Noise(object):
    def __init__(self,t,y):
        self.t = t
        self.y = y
        self.original_y = y
        self.original_y_mean = self.original_y.mean()

    def set_t(self, t):
        """Set t values
        """
        self.t = t

    def subtract_avg(self):
        """Removes the average speed from the observations
        """
        self.y = self.y - self.original_y_mean
        return self.y

    def subtract(self, array):
        self.y = self.y - array
        return self.y

    def single_component(self, A_i_FS, w_i_FS, phi_i_FS):
        """ Returns a single velocity noise component as described in the sum of eqn (5)
        """
        return A_i_FS * np.sin( (w_i_FS*self.t) + phi_i_FS )
     
    def eqn_model(self, params):
        """ Returns the velocity noise FS model as described in eqn (5)
        """
        # put all the paramters in a list
        A_FS = [params['A1_FS'],params['A2_FS'],params['A3_FS']]
        w_FS = [params['w1_FS'],params['w2_FS'],params['w3_FS']]
        phi_FS = [params['phi1_FS'],params['phi2_FS'],params['phi3_FS']]

        # equation (5), sum of 3 components
        model = self.single_component(A_FS[0],w_FS[0], phi_FS[0])
        model += self.single_component(A_FS[1],w_FS[1], phi_FS[1])
        model += self.single_component(A_FS[2],w_FS[2], phi_FS[2])

        return model

    def final_curve(self):
        """ Returns the final fitted curve
        """
        final = self.eqn_model(self.LF_params.params)
        final += self.eqn_model(self.MF_params.params)
        final += self.eqn_model(self.HF_params.params)

        return final

    def fit_all(self, LF_param = LF_Noise(), MF_param = MF_Noise(), HF_param = HF_Noise()):
        """ Fits the velicoty noise for all FS components after self.subtract_avg() has been called
        """
        # fits the velocity noise for LF components
        self.LF_fit(LF_param)
        # subtract the LF components from the velocity noise
        self.subtract(self.eqn_model(self.LF_params.params))
        # fits the velocity noise for MF components
        self.MF_fit(MF_param)
        # subtract the MF components from the velocity noise
        self.subtract(self.eqn_model(self.MF_params.params))
        # fits the velocity noise for HF components
        self.HF_fit(HF_param)

    def LF_fit(self, init_params = LF_Noise()):
        """ Fits and returns the component parameters for LF noise
        """
        self.LF_params = self.NLLSR(init_params)

        return self.LF_params
    
    def MF_fit(self, init_params = MF_Noise()):
        """ Fits and returns the component parameters for HF noise
        """
        self.MF_params = self.NLLSR(init_params)

        return self.MF_params

    def HF_fit(self, init_params = HF_Noise()):
        """ Fits and returns the component parameters for HF noise
        """
        self.HF_params = self.NLLSR(init_params)

        return self.HF_params

    def fnc2min(self, params):
        """ Returns the residuals (eqn 7) for the model for when running all 3 components at once
        """
        return (self.y - self.eqn_model(params))

    # def fnc2min(self, params):
    #     """ Returns the residuals (eqn 7) for the model for when running one component at a time
    #     """
    #     return (self.y - self.single_component(params['A_FS'], params['w_FS'], params['phi_FS']))


    def NLLSR(self, LMparams):
        """ Returns the result of the NLLSR using LMFit
        """
        # uses least swuares method to minimize the parameters given by LMparams according to the residuals given by self.fnc2min
        LMFitmin = Minimizer(self.fnc2min, LMparams)
        LMFitResult = LMFitmin.minimize(method='least_squares')
        lmfit.printfuncs.report_fit(LMFitResult.params)

        return LMFitResult

if __name__ == '__main__':
    # loads the csv file and extract the attribute informations
    file_name = 'october_21_to_31.csv'
    subdir = ''
    extract_obj = Extract_Hist(file_name, subdir)
    # get the slice of ONLY cruising period
    cruising_data = extract_obj.cruise_with_vn[270]
    # create a numpy array of just t values starting at t=1
    t = np.linspace(1,len(cruising_data),len(cruising_data))
    # create a numpy array of speed_mps values
    y = cruising_data.to_numpy()
    # interpolate linearly and make timesteps finer (0.001s)
    from scipy.interpolate import interp1d
    f = interp1d(t, y)
    t = np.linspace(1,len(cruising_data),1000*len(cruising_data))
    y = f(t)

    # initialise the VN object
    vn_obj = Velocity_Noise(t,y)
    # deduct the average from the cruising period speed values (from fig3a to fig3b) and store as y
    y = vn_obj.subtract_avg()

    original_y = y
    
    # perform NLLSR with the initial parameters suggested by LMParams
    hi = vn_obj.fit_all()
    yy = vn_obj.final_curve()
    
    pulse_duration=10000
    driving_pulse = DP(pulse_duration, extract_obj)
    driving_pulse.generate_driving_pulse(vn_obj)
    driving_pulse.generate_driving_pulse(vn_obj)
    driving_pulse.generate_driving_pulse(vn_obj)
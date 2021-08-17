import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic
import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

class CryptoDataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name = "processed_btc_data.csv", root_dir = "Data/", training_length = "48", forecast_window = "12"):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # return df size
        return self.df.shape[0]

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        # idx = idx+1

#         np.random.seed(0)
        start = np.random.randint(0, self.df.shape[0] - self.T - self.S)
        
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
#         _input = torch.tensor(self.df[["price_close", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
#         target = torch.tensor(self.df[["price_close", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)
        _input = torch.tensor(self.df[["price_close", "year", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
        target = torch.tensor(self.df[["price_close", "year","sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)        
        
        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target
    
#ToDo: create a new dataset class here. Instead of using random input index, starts from 0 to len(dataset)
    
    
    
    
class CryptoDataSetWithMovingAverage(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name = "processed_btc_data.csv", root_dir = "Data/", training_length = "48", forecast_window = "12"):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

    def __len__(self):
        # return df size
        return self.df.shape[0]

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        
        # Sensors are indexed from 1
        # idx = idx+1

        # np.random.seed(0)
        
        logging.info('df shape is {}'.format(self.df.shape))
        
        start = np.random.randint(0, self.df.shape[0] - self.T - self.S) 
        
        index_in = torch.tensor([i for i in range(start, start+self.T)])
        index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
#         _input = torch.tensor(self.df[["price_close", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
#         target = torch.tensor(self.df[["price_close", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)
        _input = torch.tensor(self.df[["price_close", "year", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start : start + self.T].values)
        target = torch.tensor(self.df[["price_close", "year","sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"]][start + self.T : start + self.T + self.S].values)
        

        # moving average: https://krzjoa.github.io/2019/12/28/pytorch-ts-v1.html
        kernel = [0.125]*8 # what is the kernel should be for rolling window size=10? Kernel should be 1/window_size
        kernel_tensor = torch.tensor(kernel).reshape(1, 8, -1) # 1-number of time series, batch_size, 8-number of features, -1-length of the time series
        logging.info('moving average kernel is {}'.format(kernel_tensor))
        logging.info('_input shape is {}'.format(_input.shape))
        logging.info('target shape is {}'.format(target.shape))
        
        _input = _input.reshape(48, 8, -1)
        target = target.reshape(24, 8, -1)
        
        logging.info('_input shape is {}'.format(_input.shape))
        logging.info('target shape is {}'.format(target.shape))
        
        torch.nn.functional.conv1d(_input, kernel_tensor)
        torch.nn.functional.conv1d(target, kernel_tensor)
        
        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return index_in, index_tar, _input, target
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
import random
import statsmodels
import warnings
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL


class PyMinMaxScaler(object):
    """
    Transforms each series separately to the range [feature_range[0], feature_range[1]].
    
    Args:
        feature_range (tuple): Min and max values for the transformed data, default (0,1)
    
    Input shape: (sample_size, series_length) or (sample_size, dim-multivar, series_length)
    Output shape: (sample_size, series_length) or (sample_size, dim-multivar, series_length)
    """
    def __init__(self, feature_range=(0, 1)): 
        self.range_min = feature_range[0]
        self.range_max = feature_range[1] 
        self.min = None
        self.max = None
        self.shape = None
        
    def fit(self, X):
        """
        Compute the min and max values for scaling
        
        Args:
            X: Input tensor to fit the scaler on
        """
        self.shape = X.shape
        if len(self.shape) == 3:
            X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        self.max = torch.max(X, dim=1)[0].reshape(-1, 1)
        self.min = torch.min(X, dim=1)[0].reshape(-1, 1)

    def transform(self, X):
        """
        Scale the data based on the fitted parameters
        
        Args:
            X: Input tensor to transform
            
        Returns:
            torch.Tensor: Scaled tensor
        """
        if X.shape != self.shape:
            raise Exception("Input shape and fitted shape do not match") 
            
        if len(self.shape) == 3:
            X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])       
        
        # Standard scaling formula: (X - min) / (max - min)
        X_std = torch.div((X - self.min), (self.max - self.min))
        is_nan = torch.isnan(X_std).any(dim=1)
        
        # Handle NaN values by replacing with range_min
        valid_ind = torch.where(is_nan == False)[0]
        X_scaled = torch.mul(X_std[valid_ind], (self.range_max - self.range_min)) + self.range_min
        
        X_output = (torch.ones(X.shape) * self.range_min).to(X.device)
        X_output[valid_ind] = X_scaled
        X_output = X_output.reshape(self.shape)
        return X_output
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it
        
        Args:
            X: Input tensor
            
        Returns:
            torch.Tensor: Scaled tensor
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Scale back the data to the original representation
        
        Args:
            X_scaled: Scaled input tensor
            
        Returns:
            torch.Tensor: Original tensor
        """
        if X_scaled.shape != self.shape:
            raise Exception("Input shape and fitted shape do not match") 
            
        if len(self.shape) == 3:
            X_scaled = X_scaled.reshape(X_scaled.shape[0] * X_scaled.shape[1], X_scaled.shape[2])   

        # Inverse scaling formula: X = (X_scaled - min_range) / (max_range - min_range) * (max - min) + min
        X = (X_scaled - self.range_min) / (self.range_max - self.range_min) * (self.max - self.min) + self.min
        X = X.reshape(self.shape)
        return X


def PySTL(data, seasonality):
    """
    Runs STL decomposition on the input, for each series separately
    
    Args:
        data: Input tensor or numpy array of shape (batch_size, series_length) 
              or (batch_size, dim-multivar, series_length)
        seasonality: Period of the seasonal component. If None, uses 20 as default
        
    Returns:
        tuple: (trend, seasonal, residual) components as torch tensors
    """
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().detach().numpy()
    else:
        data_np = data
        
    orig_shape = data.shape
    batch_size = orig_shape[0]
    
    # Handle multivariate case
    if len(orig_shape) == 3:
        series_len = orig_shape[2]
        multivar_n = orig_shape[1]
        data_np = data_np.reshape(batch_size * multivar_n, series_len)
    else:
        series_len = orig_shape[1]
        multivar_n = 1
    
    # Set seasonality parameter
    seas = 20 if seasonality is None else seasonality
    
    trend_list = []
    residual_list = []
    seasonal_list = []

    # Apply STL decomposition to each series
    for ser in data_np:
        stl = STL(ser, period=seas)
        res = stl.fit()
        trend_list.append(res.trend)
        residual_list.append(res.resid)
        seasonal_list.append(res.seasonal)
    
    # Reshape results back to original dimensions
    data_trend = np.concatenate(trend_list).reshape(batch_size * multivar_n, series_len)
    data_seasonal = np.concatenate(seasonal_list).reshape(batch_size * multivar_n, series_len)
    data_residual = np.concatenate(residual_list).reshape(batch_size * multivar_n, series_len)
    
    data_trend = torch.tensor(data_trend).reshape(orig_shape)
    data_seasonal = torch.tensor(data_seasonal).reshape(orig_shape)
    data_residual = torch.tensor(data_residual).reshape(orig_shape)
    
    # If seasonality is None, combine seasonal and residual components
    if seasonality is None:
        data_residual = data_residual + data_seasonal
        data_seasonal = data_residual
                        
    return (data_trend.float(), data_seasonal.float(), data_residual.float())


def stl_for_vec(ser, seas):
    """
    Apply STL decomposition to a single vector
    
    Args:
        ser: Input series
        seas: Seasonality period
        
    Returns:
        tuple: (trend, residual, seasonal) components
    """
    stl = STL(ser, period=seas)
    res = stl.fit()
    return res.trend, res.resid, res.seasonal


def Pybox_cox(x, lambda_):
    """
    Apply Box-Cox transformation to the data
    
    Args:
        x: Input tensor
        lambda_: Box-Cox transformation parameter
        
    Returns:
        torch.Tensor: Transformed tensor
    """
    if lambda_ != 0:
        y = (torch.pow(x, lambda_) - 1) / lambda_
    else:
        y = torch.log(x)
    return y


def Pyinv_box_cox(y, lambda_):
    """
    Apply inverse Box-Cox transformation
    
    Args:
        y: Input tensor (transformed)
        lambda_: Box-Cox transformation parameter
        
    Returns:
        torch.Tensor: Original tensor
    """
    if lambda_ != 0:
        x = torch.pow(y * lambda_ + 1, 1/lambda_)
    else:
        x = torch.exp(y)
    return x


def parse_method_to_policy(method_str):
    """
    Parse a method string into a policy list of operations and magnitudes
    
    Args:
        method_str (str): Method string in format "Operation_Value_Operation_Value..."
        Example: "Smooth_1.0_Jittering_0.5"
        
    Returns:
        list: Single list with [operation1, value1, operation2, value2, ...]
        Example: [['Smooth', 1.0, 'Jittering', 0.5]]
    """
    if not method_str:
        return []
        
    parts = method_str.split('_')
    policy = []
    
    # Process parts in pairs and add to flat list
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            operation = parts[i]
            try:
                value = float(parts[i+1])
                policy.append(operation)
                policy.append(value)
            except ValueError:
                # Skip if value cannot be converted to float
                pass
                
    return [policy]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import timeit
from autoaugment.utils import PyMinMaxScaler, PySTL


def jittering(data, mag, rng=None):
    """Add random noise to data scaled by magnitude (range [0, 0.1])"""
    if rng is None:
        rng = np.random.default_rng()
    
    scaler = PyMinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    noise = rng.normal(loc=0, scale=np.ones(data.shape) * mag)
    noise = torch.tensor(noise).to(data.device)
    
    jittered = scaler.inverse_transform(scaled_data + noise)
    return jittered


def flip(data, mag):
    """Flip the data if mag > 0"""
    if mag == 0:
        return data
    
    scaler = PyMinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    scaler_flipped = PyMinMaxScaler(feature_range=(0, 1))
    scaled_flipped = scaler_flipped.fit_transform(data * (-1))
    
    flipped = scaler.inverse_transform(scaled_flipped)
    return flipped


def trend_scale(data, mag, seasonality=12, stl_decomp=None):
    """Scale the trend component by magnitude
       - upscale [1, 10]
       - downscale [0, 1]
    """
    # Use precomputed STL if available
    if stl_decomp is not None:
        data_trend = stl_decomp[0].to(data.device)
    else:
        data_trend, _, _ = PySTL(data, seasonality)
        data_trend = data_trend.to(data.device)

    trend_scaler = PyMinMaxScaler(feature_range=(0, 1))
    data_trend_s = trend_scaler.fit_transform(data_trend)

    data_trend_s_scaled = data_trend_s * mag
    data_trend_scaled = trend_scaler.inverse_transform(data_trend_s_scaled)

    trend_scaled = data - data_trend + data_trend_scaled
    return trend_scaled


def seasonality_scale(data, mag, seasonality=12, stl_decomp=None):
    """Scale the seasonal component by magnitude
       - upscale [1, 3]
       - downscale [0, 1]
    """
    # Use precomputed STL if available
    if stl_decomp is not None:
        data_seas = stl_decomp[1].to(data.device)
    else:
        _, data_seas, _ = PySTL(data, seasonality)
        data_seas = data_seas.to(data.device)
    
    seas_scaled = data + data_seas * (mag - 1)
    return seas_scaled


def scale(data, mag):
    """Scale the entire time series by magnitude
       - upscale [1.0, 3.0]
       - downscale [0.3, 1.0]
    """
    if mag == 0:
        return data
    
    scaled = data * mag
    return scaled


def reverse(data, mag):
    """Reverse the time series if mag > 0"""
    if mag == 0:
        return data
    
    r = torch.flip(data, dims=(-1,))
    return r


def permutation(data, mag, n_per=1, rng=None):
    """Randomly permute segments of the time series
       - range [0, 0.3], where 0.3 means more segments (smaller segments)
    """
    if mag == 0:
        return data
    
    mag = int(1/mag)  # Convert to number of segments
    
    if rng is None:
        rng = np.random.default_rng()
    
    orig_shape = data.shape
    if len(orig_shape) == 3:
        data = data.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
    
    if mag >= len(data[0]):
        return data.reshape(orig_shape)

    segments = np.linspace(0, len(data[0]), mag + 1, dtype=int)
    perm = data
    
    for per in range(n_per):
        index_segment1 = rng.choice(np.arange(1, mag + 1, 1))
        index_segment2 = rng.choice(np.arange(1, mag + 1, 1))
        
        while (index_segment1 == index_segment2):
            index_segment2 = rng.choice(np.arange(1, mag + 1, 1))

        min_index = min(index_segment1, index_segment2)
        max_index = max(index_segment1, index_segment2)

        right = perm[:, segments[max_index]:]
        left = perm[:, :segments[min_index-1]]
        center = perm[:, segments[min_index]:segments[max_index-1]]
        right_per = perm[:, segments[min_index-1]:segments[min_index]]
        left_per = perm[:, segments[max_index-1]:segments[max_index]]

        perm = torch.cat([left, left_per, center, right_per, right], dim=1)
    
    perm = perm.reshape(orig_shape)
    return perm


def window_warping(data, mag):
    """Stretch or compress the time series
       - upscale [1, 1.5]: stretch
       - downscale [0.5, 1]: compress
    """
    orig_shape = data.shape
    size = int(mag * orig_shape[-1])
    r = np.linspace(0, orig_shape[-1] - 1, size)
    indices = np.round(r).astype(int)
    
    if len(orig_shape) == 3:
        warped = data[:, :, indices]
    else:
        warped = data[:, indices]
    
    return warped


def dts(data, mag, w=None, rng=None):
    """Dynamic Time Scaling - warp segments of time series differently
       - range [1, 1.5]
    """
    orig_shape = data.shape
    
    if rng is None:
        rng = np.random.default_rng()
    
    if w is None:
        n_windows = rng.choice([4, 5, 6])
        w = orig_shape[-1] // n_windows

    min_scale = 1/mag
    max_scale = mag

    # For each window
    for i in range(orig_shape[-1] // w + 1):
        if len(orig_shape) == 3:
            window = data[:, :, i*w:(i+1)*w]
        else:
            window = data[:, i*w:(i+1)*w]

        # Generate random scale factor
        eps = rng.uniform()
        w_mag = rng.uniform(1, max_scale) if eps > 0.5 else rng.uniform(min_scale, 1)
        window = window_warping(window, w_mag)

        if i == 0:
            warped = window
        else:
            warped = torch.cat([warped, window], dim=len(orig_shape)-1)
    
    return warped


def low_pass(data, mag):
    """Apply low-pass filter to data (smoothing)
       - range [0.0, 11.0] (k value)
    """
    k = int(mag)
    if k == 0:
        return data
    
    orig_shape = data.shape
    if len(orig_shape) == 3:
        data = data.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
    
    # Add padding
    inputs_plus = torch.cat([data[:, :k//2], data, data[:, -k//2:]], dim=1)
    inputs_plus = inputs_plus.unsqueeze(1).to(data.device)
    
    # Create geometric kernel
    space = np.geomspace(1, 10, k//2 + 1)
    hist = np.concatenate([space, np.flip(space)[1:]])
    hist_norm = hist / np.sum(hist)
    
    filters = torch.tensor(hist_norm).to(data.device)
    filters = filters.reshape(1, 1, len(filters)).float().to(data.device)
    
    # Apply convolution and remove padding
    smoothed = F.conv1d(inputs_plus, filters, padding='same')
    smoothed = smoothed[:, :, k//2:]
    smoothed = smoothed[:, :, :-k//2].squeeze(1)
    
    smoothed = smoothed.reshape(orig_shape)
    return smoothed


def high_pass(data, mag, d=2):
    """Apply high-pass filter to data (emphasize noise/details)
       - range [0, 1.0]
    """
    orig_shape = data.shape
    if len(orig_shape) == 3:
        data = data.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
    
    # Select filter based on derivative order
    if d == 1:       
        filters = torch.tensor([1, -1]).to(data.device)
        k = 2
    elif d == 2:
        filters = torch.tensor([-1, 2, -1]).to(data.device)
        k = 3
    else:
        raise ValueError("Only first and second order allowed, d > 2 is not compatible")
    
    # Add padding
    inputs_plus = torch.cat([data[:, :k//2], data, data[:, -k//2:]], dim=1)
    inputs_plus = inputs_plus.unsqueeze(1)
    
    filters = filters.reshape(1, 1, len(filters)).float()
    
    # Apply convolution and remove padding
    noise = F.conv1d(inputs_plus, filters, padding='same')
    noise = noise[:, :, k//2:]
    noise = noise[:, :, :-k//2].squeeze(1)
    
    noise_scaled = data + mag * noise.squeeze(1)
    noise_scaled = noise_scaled.reshape(orig_shape)
    return noise_scaled


def mixup(data, mag, rng=None):
    """Mix samples within the batch
       - range [0, 0.5]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    correspond_ind = rng.integers(0, len(data), size=len(data))
    mixuped = (1 - mag) * data + mag * data[correspond_ind]
    
    return mixuped


def identity(data, mag):
    """Return data unchanged"""
    return data


class AugmentTS:
    """Time series augmentation class with various transformation methods"""
    
    def __init__(self, seasonality, aug_params=None, transformations=None, seed=42):
        """
        Initialize the augmentation class
        
        Args:
            seasonality: The seasonality period for seasonal decomposition
            aug_params: Dictionary of augmentation parameters
            transformations: Dictionary of transformation functions
            seed: Random seed
        """
        self.seasonality = seasonality
        self.aug_params = aug_params

        if self.aug_params is None:
            self.aug_params = {
                'mag_bins': -1,  # -1 for continuous with range [0,1], else mag bins
            }

        if transformations:
            self.transformations = transformations
        else:
            self.transformations = {  
                "Jittering": {"range": (0.0, 0.1), "func": jittering},
                "Trend Upscale": {"range": (1.0, 10.0), "func": trend_scale},
                "Trend Downscale": {"range": (0.0, 1.0), "func": trend_scale},
                "Seasonality Upscale": {"range": (1.0, 3.0), "func": seasonality_scale},
                "Seasonality Downscale": {"range": (0.0, 1.0), "func": seasonality_scale},
                "Upscale": {"range": (1.0, 3), "func": scale},
                "Downscale": {"range": (0.3, 1.0), "func": scale},
                "Smooth": {"range": (0.0, 11.0), "func": low_pass},
                "Noise Scale": {"range": (0.0, 1.0), "func": high_pass},
                "Flip": {"range": None, "func": flip},
                "Permutation": {"range": (0.0, 0.3), "func": permutation},
                "Reverse": {"range": None, "func": reverse},
                "DTS": {"range": (1.0, 5.0), "func": dts},
                "Window Upsize": {"range": (1.0, 1.5), "func": window_warping},
                "Window Downsize": {"range": (0.5, 1.0), "func": window_warping},
                "Mixup": {"range": (0.0, 0.5), "func": mixup},
                "Identity": {"range": None, "func": identity}
            }

        self.transformation_names = list(self.transformations.keys())
        self.mag_bins = self.aug_params["mag_bins"]
        self.discrete_mag = self.mag_bins > 0
        self.policy = None
        self.rng = np.random.default_rng(seed=seed)

    def assign_policy(self, policy):
        """Set the augmentation policy"""
        self.policy = policy

    def apply_op(self, data, op_name, mag, stl_decomp=None):
        """
        Apply a specific transformation operation to the data
        
        Args:
            data: Input time series data
            op_name: Name of the transformation to apply
            mag: Magnitude of the transformation 
            stl_decomp: Optional precomputed STL decomposition
            
        Returns:
            Transformed data
        """
        if op_name not in self.transformation_names:
            raise ValueError(f"Transformation {op_name} not listed")
        
        range_vals = self.transformations[op_name]["range"]
        if range_vals is not None:
            min_range, max_range = range_vals
            
            # Handle discrete or continuous magnitude values
            if self.discrete_mag:
                dis_range = np.linspace(min_range, max_range, self.mag_bins)
                # Flip range for downscaling operations
                if op_name in ["Trend Downscale", "Seasonality Downscale", "Downscale", "Window Downsize"]:
                    dis_range = np.flip(dis_range)
                if mag == -1:
                    mag = self.rng.choice(dis_range)
                else:
                    mag = dis_range[mag]
            else:
                # Handle downscaling operations
                if op_name in ["Trend Downscale", "Seasonality Downscale", "Downscale", "Window Downsize"]:
                    temp = min_range
                    min_range = max_range
                    max_range = temp
                    
                if mag == -1:
                    mag = self.rng.uniform()                   
                mag = min_range + (max_range - min_range) * mag
                mag = np.round(mag, 3)

        if mag < 0:
            raise ValueError(f"Magnitude should be positive, got {mag}")

        # Apply the appropriate transformation
        if op_name in ["Trend Downscale", "Trend Upscale", "Seasonality Downscale", "Seasonality Upscale"]:
            output = self.transformations[op_name]["func"](
                data, mag, self.seasonality, stl_decomp)
        elif op_name in ["Mixup", "Permutation", "DTS", "Jittering"]:
            output = self.transformations[op_name]["func"](data, mag, rng=self.rng)
        else:
            output = self.transformations[op_name]["func"](data, mag)

        # Check for NaN values
        if torch.isnan(output.float()).any():
            print(f"NaN values detected in {op_name} with magnitude {mag}")
            
        return output.float()

    def iter_augment(self, data, prob=None, stl_decomp=None):
        """
        Apply augmentation policy to the data
        
        Args:
            data: Input time series data
            prob: Optional probability for applying transformations
            stl_decomp: Optional precomputed STL decomposition
            
        Returns:
            Augmented data
        """
        batch_size = data.shape[0]
        orig_len = data.shape[-1]

        # Handle multivariate case
        if len(data.shape) == 3:
            multivar_n = data.shape[1]
            aug_data = data.clone()
        else:
            multivar_n = 1
            aug_data = data.clone().unsqueeze(1)
            if stl_decomp is not None:
                stl_decomp = stl_decomp.unsqueeze(2)

        # Track time spent on each transformation
        time_m = dict.fromkeys(self.transformation_names, 0)
        
        # Randomly select policy for each example
        pol_indx = self.rng.choice(len(self.policy), size=batch_size)
        pol_per_example = pd.Series(pol_indx)
        
        # Apply selected policies
        for i_pol in range(len(self.policy)):
            indx1 = pol_per_example[pol_per_example == i_pol].index.values
            sub_policy = self.policy[i_pol] 
            
            # Each sub-policy consists of pairs: (transformation, magnitude)
            for i_op in range(len(sub_policy) // 2):              
                transformation = sub_policy[i_op * 2]
                if transformation not in self.transformation_names + ['uniform']:
                    raise ValueError(f"The transformation {transformation} does not exist.")
                
                mag = sub_policy[i_op * 2 + 1]
                
                # Handle uniform random selection of transformation
                if transformation == "uniform":
                    tr_indx = self.rng.choice(self.transformation_names, size=len(indx1))
                    tr_per_example = pd.Series(data=tr_indx, index=indx1)
                else:
                    tr_per_example = pd.Series(data=[transformation] * len(indx1), index=indx1)
                
                # Apply each transformation to the relevant examples
                for tr in self.transformation_names:
                    start = timeit.default_timer()
                    indx2 = tr_per_example[tr_per_example == tr].index.values
                    
                    if len(indx2) == 0:
                        continue
                        
                    tr_data = aug_data[indx2].clone()
                    tr_stl_decomp = stl_decomp[:, indx2].clone() if stl_decomp is not None else None
                    
                    # Handle transformations that may change the length
                    if tr in ["DTS", "Window Downsize", "Window Upsize", "Reverse", "Permutation"]:
                        tr_data = self.apply_op(tr_data, tr, mag)
                        dif = orig_len - tr_data.shape[-1]
                        
                        # Adjust length if needed
                        if dif > 0:
                            tr_data = torch.cat(
                                [torch.zeros((len(tr_data), multivar_n, dif), device=data.device), tr_data], 
                                dim=2
                            )
                        if dif < 0:
                            tr_data = tr_data[:, :, -dif:]
                            
                        aug_data[indx2] = tr_data
                    else:
                        aug_data[indx2] = self.apply_op(tr_data, tr, mag, tr_stl_decomp)
                        
                    stop = timeit.default_timer()
                    time_m[tr] += stop - start
                    
        # Restore original shape if needed
        if len(data.shape) < 3:
            aug_data = aug_data.squeeze(1)
            
        return aug_data

    def augment(self, batch, horizon, prob=None):
        """
        Main entry point for augmentation
        
        Args:
            batch: Dictionary containing data and optionally decomposition
            horizon: Forecast horizon length
            prob: Optional probability for applying transformations
            
        Returns:
            Augmented data
        """
        data = batch['data'].clone()
        
        # If no policy is set, return original data
        if self.policy is None:
            return data

        batch_size = len(data)
        ser_len = data.shape[1]
        aug_indx = np.arange(batch_size)

        # Check if STL decomposition is available in the batch
        if {'trend', 'seasonality'}.issubset(set(batch.keys())):
            trend = batch['trend'].clone()
            seas = batch['seasonality'].clone()
            
            if len(data.shape) == 3:
                trend = trend.permute(0, 2, 1)
                seas = seas.permute(0, 2, 1)
                
            stl_batch = [
                trend.unsqueeze(0),
                seas.unsqueeze(0),
            ]                 
            stl_decomp = torch.cat(stl_batch, axis=0).to(data.device)
        else:
            stl_decomp = None            

        # Handle data reshaping for multivariate case
        if len(data.shape) == 3:
            data = data.permute(0, 2, 1)
            data = data[aug_indx, :, :ser_len]
            if stl_decomp is not None:
                stl_decomp = stl_decomp[:, aug_indx, :, :ser_len]
        else:
            data = data[aug_indx, :ser_len]
            if stl_decomp is not None:
                stl_decomp = stl_decomp[:, aug_indx, :ser_len]
        
        output_data = data.clone()
        aug_data = self.iter_augment(data, prob, stl_decomp)
        output_data[aug_indx] = aug_data
        
        # Restore original shape before returning
        if len(output_data.shape) == 3:
            output_data = output_data.permute(0, 2, 1)

        return output_data
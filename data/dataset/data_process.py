import torch
import numpy as np
import pandas as pd

class ScalarStandardizer:
    """
    A PyTorch-based scalar standardizer for standardizing data (mean=0, std=1).
    Suitable for both CPU and GPU operations.
    """
    def __init__(self, device='cpu'):
        self.mean = None
        self.std = None
        self.device = device

    def _to_tensor(self, data):
        """
        Convert input data to a PyTorch Tensor if it's a pandas DataFrame or numpy array.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Converted PyTorch Tensor.
        """
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values, dtype=torch.float32, device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise ValueError("Unsupported data type. Must be a numpy array, pandas DataFrame, or PyTorch Tensor.")

    def fit(self, data):
        """
        Compute the mean and standard deviation from the given data.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).
        """
        data = self._to_tensor(data)
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)

    def transform(self, data):
        """
        Standardize the data using the computed mean and std.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("ScalarStandardizer requires fitting before transforming.")

        data = self._to_tensor(data)
        return (data - self.mean) / (self.std + 1e-8)  # Add epsilon to avoid division by zero

    def fit_transform(self, data):
        """
        Fit the standardizer to the data, then standardize it.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Standardized data.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Reverse the standardization process to recover original data.

        Args:
            data: Standardized data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Original data.
        """
        if self.mean is None or self.std is None:
            raise ValueError("ScalarStandardizer requires fitting before inverse transforming.")

        data = self._to_tensor(data)
        return data * (self.std + 1e-8) + self.mean

class MinMaxScaler:
    """
    A PyTorch-based MinMax scaler for scaling data to a specified range [min, max].
    Suitable for both CPU and GPU operations.
    """
    def __init__(self, feature_range=(0, 1), device='cuda'):
        self.min_val = None
        self.max_val = None
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.device = device

        if self.min >= self.max:
            raise ValueError("Minimum of feature range must be less than maximum.")

    def _to_tensor(self, data):
        """
        Convert input data to a PyTorch Tensor if it's a pandas DataFrame or numpy array.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Converted PyTorch Tensor.
        """
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values, dtype=torch.float32, device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise ValueError("Unsupported data type. Must be a numpy array, pandas DataFrame, or PyTorch Tensor.")

    def fit(self, data):
        """
        Compute the minimum and maximum values from the given data.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).
        """
        data = self._to_tensor(data)
        self.min_val = data.min(dim=0, keepdim=True).values
        self.max_val = data.max(dim=0, keepdim=True).values

    def transform(self, data):
        """
        Scale the data to the specified range [min, max].

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Scaled data.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("MinMaxScaler requires fitting before transforming.")

        data = self._to_tensor(data)
        scaled_data = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)  # Avoid division by zero
        return scaled_data * (self.max - self.min) + self.min

    def fit_transform(self, data):
        """
        Fit the scaler to the data, then scale it.

        Args:
            data: Input data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Scaled data.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Reverse the scaling process to recover original data.

        Args:
            data: Scaled data (numpy array, pandas DataFrame, or PyTorch Tensor).

        Returns:
            torch.Tensor: Original data.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("MinMaxScaler requires fitting before inverse transforming.")

        data = self._to_tensor(data)
        original_data = (data - self.min) / (self.max - self.min + 1e-8)  # Normalize to [0, 1]
        return original_data * (self.max_val - self.min_val) + self.min_val

# Example usage
if __name__ == "__main__":
    # Generate sample data
    data = np.random.randn(100, 10)  # 100 samples, 10 features

    # Create and use the standardizer
    standardizer = ScalarStandardizer()
    standardized_data = standardizer.fit_transform(data)

    # Recover original data
    original_data = standardizer.inverse_transform(standardized_data)

    # Verify the process
    print("Original Mean:", torch.tensor(data).mean(dim=0))
    print("Standardized Mean:", standardized_data.mean(dim=0))
    print("Recovered Data Mean:", original_data.mean(dim=0))
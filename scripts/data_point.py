import numpy as np
from uncertainties import unumpy, ufloat, core

class DataPoint:
    # A data point (or an array of data points) with both statistical and systematic uncertainties.
    # These uncertainties are considered independent.
    # Propagate statistical and systematic uncertainties separately using two uncertainties variables.
    value = None
    stat_error = None
    sys_error = None
    def __init__(self, value, stat_error=None, sys_error=None):
        """
        Initialize a DataPoint object.
        
        Parameters:
        value (float or np.ndarray): The value of the data point.
        stat_error (float or np.ndarray, optional): The statistical uncertainty. Defaults to None.
        sys_error (float or np.ndarray, optional): The systematic uncertainty. Defaults to None.
        """
        if isinstance(value, np.ndarray):
            self.value = value
        else:
            self.value = np.array([value])
        
        if stat_error is None:
            self.stat_error = np.zeros_like(self.value)
        elif isinstance(stat_error, np.ndarray):
            self.stat_error = stat_error
        else:
            self.stat_error = np.array([stat_error])
        
        if sys_error is None:
            self.sys_error = np.zeros_like(self.value)
        elif isinstance(sys_error, np.ndarray):
            self.sys_error = sys_error
        else:
            self.sys_error = np.array([sys_error])

    def __repr__(self):
        return f"DataPoint(value={self.value}, stat_error={self.stat_error}, sys_error={self.sys_error})"
    
    def __add__(self, other):
        if not isinstance(other, DataPoint):
            raise ValueError("Can only add another DataPoint.")
        
        if len(self.value) != len(other.value):
            raise ValueError("DataPoints must have the same length.")
        
        new_val = self.value + other.value
        new_stat_err = np.sqrt(self.stat_error**2 + other.stat_error**2)
        new_sys_err = np.sqrt(self.sys_error**2 + other.sys_error**2)
        
        return DataPoint(new_val, new_stat_err, new_sys_err)
    
    def __sub__(self, other):
        if not isinstance(other, DataPoint):
            raise ValueError("Can only subtract another DataPoint.")
        
        if len(self.value) != len(other.value):
            raise ValueError("DataPoints must have the same length.")
        
        new_val = self.value - other.value
        new_stat_err = np.sqrt(self.stat_error**2 + other.stat_error**2)
        new_sys_err = np.sqrt(self.sys_error**2 + other.sys_error**2)
        
        return DataPoint(new_val, new_stat_err, new_sys_err)
    
    def __mul__(self, other):
        if not isinstance(other, (int, float, DataPoint)):
            raise ValueError("Can only multiply by a scalar or another DataPoint.")
        
        if isinstance(other, DataPoint):
            if len(self.value) != len(other.value):
                raise ValueError("DataPoints must have the same length.")
            new_val = self.value * other.value
            # error propagation for z = x*y, stat and sys propagated separately:
            # (Dz)^2 = (y*Dx)^2 + (x*Dy)^2
            new_stat_err = np.sqrt((self.stat_error * other.value)**2 + (other.stat_error * self.value)**2)
            new_sys_err = np.sqrt((self.sys_error * other.value)**2 + (other.sys_error * self.value)**2)
        else:
            new_val = self.value * other
            new_stat_err = self.stat_error * abs(other)
            new_sys_err = self.sys_error * abs(other)
        
        return DataPoint(new_val, new_stat_err, new_sys_err)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, index):
        """
        Get a specific index of the DataPoint.
        
        Parameters:
        index (int or slice): The index or slice to access the DataPoint.
        
        Returns:
        DataPoint: A new DataPoint object with the value, stat_error, and sys_error at the specified index or slice.
        """
        return DataPoint(self.value[index], self.stat_error[index], self.sys_error[index])
    
    def add_point(self, value, stat_error=None, sys_error=None):
        """
        Add a new point to the DataPoint.
        
        Parameters:
        value (float): The value of the new point.
        stat_error (float, optional): The statistical uncertainty of the new point. Defaults to None.
        sys_error (float, optional): The systematic uncertainty of the new point. Defaults to None.
        """
        self.value = np.append(self.value, value)
        self.stat_error = np.append(self.stat_error, stat_error if stat_error is not None else 0)
        self.sys_error = np.append(self.sys_error, sys_error if sys_error is not None else 0)
    
    def total_error(self):
        """
        Calculate the total uncertainty of the data point.
        
        Returns:
        np.ndarray: The total uncertainty, which is the square root of the sum of squares of statistical and systematic uncertainties.
        """
        return np.sqrt(self.stat_error**2 + self.sys_error**2)
    
    def average(self):
        """
        Calculate the average value of the data point. Use the inverse stat error squared as weights.

        Returns:
        ufloat: The average value with its total uncertainty (statistical and systematic
                combined in quadrature).
        """
        weights = 1 / self.stat_error**2
        mean = np.average(self.value, weights=weights)
        stat_err = np.sqrt(1 / np.sum(weights))
        # systematic error of the weighted mean: per-point sys errors propagated as
        # independent contributions through v_bar = sum(w_i v_i) / sum(w_i)
        sys_err = np.sqrt(np.sum((weights * self.sys_error) ** 2)) / np.sum(weights)
        return ufloat(mean, np.sqrt(stat_err ** 2 + sys_err ** 2))
        
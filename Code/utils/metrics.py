import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

def last_timestep_mse(y_true, y_pred):
    return MeanSquaredError()(y_true[:,-1,:], y_pred[:,-1,:])

def last_timestep_mae(y_true, y_pred):
    return MeanAbsoluteError()(y_true[:,-1,:], y_pred[:,-1,:])

def metrics (y_true,y_pred,filtON = False):
    """Calculates metrics for regressive modeling.
    Args:
        y_true: Real target values (array).
        y_pred: Predicted target values (array).
        filtON: If True, the function filters results (butterworth filter).
                You must pass the array as np.hstack(). 
    Returns:
        dic_metrics: Dictionary with metrics.
                    - R2 Score
                    - MAE
                    - MSE
                    - Correlation
                    - Shift (Positive Correlation)
                    - Shift (Negative Correlation)
    """
    def positive_shift_corr(y_true,y_pred):
        """Metric that advances the real signal and compares it with the current modeling.
        Args:
            y_true: Real target values.
            y_pred: Predicted target values.
        Return:
            k_best: The bigger it means that a prediction is happening with that lag.
            corr_max: The bigger it means that the prediction highly correlated.
        """
        k_best,corr_max = 1,0.0
        for k in np.arange(1,50,3):
            corr_actual = np.corrcoef((y_true[k:],y_pred[:-k]))[1,0]
            if (corr_actual >=corr_max):
                k_best,corr_max = k, corr_actual
    
        return (k_best,np.round(corr_max,2))
    
    def negative_shift_corr(y_true,y_pred):
        """Metric that delays the real signal and compares with the current modeling.
        Args:
            y_true: Real target values.
            y_pred: Predicted target values.
        Return:
            k_best: Lag value.
            corr_max: Max Correlation.  
        If the correlation value is high, it means that it may be repeating the value of K previous samples of the signal.
        The lower the correlation value, and the higher the K value, the better. 
        """

        k_best,corr_max = 1,0.0
        for k in np.arange(1,50,3):
            corr_actual = np.corrcoef((y_true[:-k],y_pred[k:]))[1,0]
            if (corr_actual >=corr_max):
                k_best,corr_max = k, corr_actual
                
        return (k_best,np.round(corr_max,2))
    
    last_y_true, last_y_pred = y_true[:,-1,:], y_pred[:,-1,:]
    dic_metrics = {}
    dic_metrics['r2']  = r2_score(last_y_true,last_y_pred)
    dic_metrics['mae'] = mean_absolute_error(last_y_true,last_y_pred)
    dic_metrics['mse'] = mean_squared_error(last_y_true,last_y_pred)
    dic_metrics['corr']  = np.corrcoef((last_y_true.flatten(),last_y_pred.flatten()))[1,0]
    dic_metrics["positive_shift"] = [positive_shift_corr(last_y_true.flatten(),last_y_pred.flatten())]
    dic_metrics["negative_shift"] = [negative_shift_corr(last_y_true.flatten(),last_y_pred.flatten())]
    
    df_percentil = pd.DataFrame(np.abs(last_y_true-last_y_pred))
    dic_metrics['25%'] = df_percentil.quantile(0.25)
    dic_metrics['50%'] = df_percentil.quantile(0.50)
    dic_metrics['75%'] = df_percentil.quantile(0.75)
    
    return dic_metrics

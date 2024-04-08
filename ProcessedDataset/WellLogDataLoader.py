from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

def slidewindow(normalised_logs, depths, observation_mask, window_size= 500, step = 100):
    for log_ind in range(len(normalised_logs)):
        assert len(normalised_logs[log_ind]) == len(depths[log_ind])
        assert len(normalised_logs[log_ind]) == len(observation_mask[log_ind])
        log = normalised_logs[log_ind]
        depth = depths[log_ind]
        mask = observation_mask[log_ind]
        for i in range(0, len(log), step):
            if i + window_size > len(log):
                break
            windowed_logs = log[i:i+window_size]
            windowed_depths = depth[i:i+window_size]
            windowed_mask = mask[i:i+window_size]
            yield windowed_logs, windowed_depths, windowed_mask

class WellLogDataLoader(Dataset):
    def __init__(self, train_test_or_valid, specific_log_path = None, window_size= 500, step = 100):
        logs_save_path = f'./ProcessedDataset/{train_test_or_valid}/'
        
        # Load StandardScaler
        if not os.path.exists('./ProcessedDataset/StandardScaler.pkl'):
            train_log_save_path = os.path.join('./ProcessedDataset/train')
            train_logs = [pd.read_csv(os.path.join(train_log_save_path, log)) for log in os.listdir(train_log_save_path)]
            train_logs = pd.concat(train_logs, axis=0).drop(['DEPT', 'DPOR'], axis=1)
            self.scaler = StandardScaler().fit(train_logs)
            joblib.dump(self.scaler, './ProcessedDataset/StandardScaler.pkl')
        else:
            self.scaler = joblib.load('./ProcessedDataset/StandardScaler.pkl')
        
        if specific_log_path is None:
            raw_logs = [pd.read_csv(os.path.join(logs_save_path, log), index_col=0).drop('DPOR', axis = 1) for log in os.listdir(logs_save_path) if log.endswith('.csv')]
            self.depths = [np.array(log.index) for log in raw_logs]
            self.normalised_logs = [self.scaler.transform(log) for log in raw_logs]
            self.observation_mask = [np.where(np.isnan(log), 0, 1) for log in raw_logs]
            self.data = [log for log in slidewindow(self.normalised_logs, self.depths, self.observation_mask, window_size=window_size, step = step)]
            self.columns = raw_logs[0].columns
        else:
            raw_logs = [pd.read_csv(specific_log_path, index_col=0).drop('DPOR', axis = 1)]
            self.depths = [np.array(log.index) for log in raw_logs]
            self.normalised_logs = [self.scaler.transform(log) for log in raw_logs]
            self.observation_mask = [np.where(np.isnan(log), 0, 1) for log in raw_logs]
            self.data = [log for log in slidewindow(self.normalised_logs, self.depths, self.observation_mask, window_size= 500, step = 500)]
            self.columns = raw_logs[0].columns


    def __getitem__(self, index):
        log, depth, mask = self.data[index]

        return {'observed_log': torch.from_numpy(np.nan_to_num(log, False)), 
                'depth_info': torch.from_numpy(depth), 
                'observed_mask': torch.from_numpy(mask)
                }

    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = WellLogDataLoader('train')
    import seaborn as sns
    print(len(dataset))
    
    for i in range(len(dataset)):
        observed_log, depth_info, observed_mask = dataset[i]['observed_log'], dataset[i]['depth_info'], dataset[i]['observed_mask']
        observed_log = observed_log.numpy()
        depth_info = depth_info.numpy()
        observed_mask = observed_mask.numpy()
        plt.figure(figsize=(15, 7))
        sns.set_style("whitegrid")
    # 创建多个子图
        for i in range(len(dataset.columns)):
            ax = plt.subplot(1, len(dataset.columns), i + 1)
            ax.plot(observed_log[:, i], depth_info, color='red')
            ax.set_title(dataset.columns[i])
            ax.set_yticklabels([])
            ax.get_yaxis().set_ticks([])  # Removes the y-axis tick marks as well
        plt.show()
        plt.close()
        
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from natsort import natsorted
import mne
from pyts.image import RecurrencePlot


class SleepDataset(Dataset):
    def __init__(self, data_folder_path, label_folder_path, save_path=None, mode=None, fs=200, filter_params=None):
        """
        SleepDataset 数据集加载器，支持 MNE 降噪处理。

        参数:
            data_folder_path (str): 特征数据文件夹路径。
            label_folder_path (str): 标签数据文件夹路径。
            save_path (str): 合并后的文件保存路径。
            mode (str): 加载模式，可选 'raw', 'spectrogram', 'recurrence_plot', 或 'combined'。
            fs (int): 采样频率。
            filter_params (dict): 滤波参数，包括 'l_freq' (低截止频率) 和 'h_freq' (高截止频率)。
        """
        self.data_folder_path = data_folder_path
        self.label_folder_path = label_folder_path
        self.save_path = save_path or 'merged_sleep_data.npz'
        self.mode = mode
        self.fs = fs
        self.filter_params = filter_params or {"l_freq": 0.5, "h_freq": 50}

        # 如果已经存在合并的文件，直接加载
        if os.path.exists(self.save_path):
            loaded_data = np.load(self.save_path, allow_pickle=True)
            self.data = loaded_data['data']
            self.labels = loaded_data['labels']

        else:
            # 获取文件名并排序
            self.npy_files = [f for f in os.listdir(data_folder_path) if f.endswith('.npy')]
            self.txt_files = [f for f in os.listdir(label_folder_path) if f.endswith('.txt')]
            self.npy_files = natsorted(self.npy_files)
            self.txt_files = natsorted(self.txt_files)

            data_list = []
            label_list = []

            # 加载所有样本和标签
            for npy_file, txt_file in zip(self.npy_files, self.txt_files):
                # 加载特征数据
                data = np.load(os.path.join(self.data_folder_path, npy_file))
                # 降噪处理
                data = self._apply_mne_filter(data)

                # 加载标签数据
                with open(os.path.join(self.label_folder_path, txt_file), 'r') as file:
                    labels = [int(line.strip()) for line in file.readlines()]

                # 转换标签 5 为 4
                labels = [4 if label == 5 else label for label in labels]

                # 确保特征数据和标签数量一致
                assert len(data) == len(labels), f"数据和标签长度不一致: {npy_file}, {txt_file}"

                # 添加数据和标签
                data_list.append(data)
                label_list.extend(labels)

            # 合并所有数据
            self.raw_data = np.vstack(data_list)  # 原始1D数据
            self.labels = np.array(label_list)

            # 根据模式生成数据
            if mode == "spectrogram":
                self.data = self._generate_spectrograms(self.raw_data)
            elif mode == "recurrence_plot":
                self.data = self._generate_recurrence_plots(self.raw_data)
            elif mode == "combined":
                # 先生成时频图和 Recurrence Plot
                spectrograms = self._generate_spectrograms(self.raw_data)
                recurrence_plots = self._generate_recurrence_plots(self.raw_data)
                # 将它们传递到 _combine_features
                self.data = self._combine_features(spectrograms, recurrence_plots)
            else:
                self.data = self.raw_data

            # 数据标准化
            self.data = (self.data - np.mean(self.data, axis=(1, 2), keepdims=True)) / np.std(self.data, axis=(1, 2), keepdims=True)

            # 保存合并后的数据
            np.savez(self.save_path, data=self.data, labels=self.labels)

        # 检查标签范围
        print(f"标签分布: {np.unique(self.labels, return_counts=True)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _apply_mne_filter(self, data):
        """
        使用 MNE 提供的带通滤波对数据降噪。

        参数:
            data (np.ndarray): 输入的 2D 数据 (num_epochs, signal_length)。
        返回:
            filtered_data (np.ndarray): 滤波后的数据。
        """
        filtered_data = []
        for epoch in data:
            # 使用 mne.filter.filter_data 进行带通滤波
            filtered_epoch = mne.filter.filter_data(epoch, sfreq=self.fs,
                                                    l_freq=self.filter_params['l_freq'],
                                                    h_freq=self.filter_params['h_freq'],
                                                    verbose=False)
            filtered_data.append(filtered_epoch)
        return np.array(filtered_data)

    def _generate_spectrograms(self, raw_data, target_shape=(64, 64)):
        """
        生成指定大小的时频图。

        参数:
            raw_data (np.ndarray): 输入的1D特征数据。
            target_shape (tuple): 输出图像的目标形状 (height, width)。
        返回:
            spectrograms (np.ndarray): 调整大小的时频图。
        """
        target_height, target_width = target_shape
        spectrograms = []
        for signal in raw_data:
            nperseg = min(len(signal) // target_width * 2, len(signal))  # 确保 nperseg <= 信号长度
            noverlap = max(0, nperseg // 2)  # 默认重叠50%，避免信号过小出错

            _, _, Zxx = stft(signal, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
            spectrogram = np.abs(Zxx)

            # 如果形状不足，补齐到 target_shape
            spectrogram = spectrogram[:target_height, :target_width]
            spectrograms.append(spectrogram)
        return np.array(spectrograms)

    def _generate_recurrence_plots(self, raw_data, target_shape=(64, 64)):
        """
        生成指定大小的 Recurrence Plot 图像。

        参数:
            raw_data (np.ndarray): 输入的1D特征数据。
            target_shape (tuple): 输出图像的目标形状 (height, width)。
        返回:
            recurrence_plots (np.ndarray): 调整大小的 Recurrence Plot 图像。
        """
        target_size = min(target_shape)  # 确保 Recurrence Plot 是方阵
        recurrence_plots = []
        for signal in raw_data:
            # 根据目标大小动态调整下采样因子
            downsample_factor = max(1, len(signal) // target_size)  # 确保因子 >= 1
            downsampled_signal = signal[::downsample_factor][:target_size]

            # RP 转换
            rp_transformer = RecurrencePlot(threshold='point', dimension=1)
            rp_image = rp_transformer.fit_transform(downsampled_signal.reshape(1, -1))[0]

            # 补齐到目标大小
            rp_image = rp_image[:target_size, :target_size]
            recurrence_plots.append(rp_image)
        return np.array(recurrence_plots)

    def _combine_features(self, spectrograms, recurrence_plots):
        """
        将时频图和 Recurrence Plot 图像在通道维度上组合。

        参数:
            spectrograms (np.ndarray): 时频图数据，形状为 [N, H, W]。
            recurrence_plots (np.ndarray): Recurrence Plot 数据，形状为 [N, H, W]。
        返回:
            combined_images (np.ndarray): 组合后的图像数据，形状为 [N, H, W, 2]。
        """
        # 检查形状一致性
        assert spectrograms.shape == recurrence_plots.shape, "两个图像的形状必须一致"

        # 在通道维度上组合
        combined_images = np.stack([spectrograms, recurrence_plots], axis=-1)  # [N, H, W, 2]
        return combined_images


if __name__ == "__main__":
    dataset = SleepDataset(
        data_folder_path='data_reading/rec_data',
        label_folder_path='data_reading/label',
        save_path='merged_sleep_data_combined.npz',
        mode="combined",  # 加载结合特征
        filter_params={"l_freq": 0.5, "h_freq": 50}
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.permute(0, 3, 1, 2)
        print(f"Batch data shape: {batch_data.shape}")  # 输出 (B, C, H, W)
        print(f"Batch labels shape: {batch_labels.shape}")
        break

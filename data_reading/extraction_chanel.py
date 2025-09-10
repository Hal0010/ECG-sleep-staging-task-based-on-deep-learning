import os
import mne
import numpy as np


def convert_edf_to_npy(input_folder, output_folder, channel_name="X2", fallback_channel="25", frame_length=6000, remove_last_frames=30):
    """
    将指定文件夹中的所有 .rec 文件（先改为 .edf）转换为 .npy 文件，提取指定通道数据，并进行重塑。

    参数：
        input_folder (str): 存放 .rec/.edf 文件的输入文件夹路径。
        output_folder (str): 保存 .npy 文件的目标文件夹路径。
        channel_name (str): 要提取的通道名称，默认 "X2"。
        fallback_channel (str): 如果主通道不存在，备选的通道名称，默认 "25"。
        frame_length (int): 每帧的长度，用于重塑数据，默认 6000。
        remove_last_frames (int): 删除最后多少帧数据，默认删除 30 帧。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件，并将 .rec 文件改为 .edf
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".rec"):
            rec_path = os.path.join(input_folder, file_name)
            edf_path = os.path.join(input_folder, file_name.replace(".rec", ".edf"))
            os.rename(rec_path, edf_path)
            print(f"Renamed {file_name} to {edf_path}")

    # 再次遍历输入文件夹中的所有 .edf 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".edf"):
            edf_path = os.path.join(input_folder, file_name)

            # 构造 .npy 文件名
            npy_file_name = file_name.replace(".edf", ".npy")
            npy_path = os.path.join(output_folder, npy_file_name)

            try:
                # 加载 .edf 文件
                raw_data = mne.io.read_raw_edf(edf_path, preload=True)
                print(f"Channels in {file_name}: {raw_data.info['ch_names']}")

                # 优先尝试提取 channel_name 通道数据，如果没有则使用 fallback_channel
                if channel_name in raw_data.info['ch_names']:
                    # 提取指定通道数据
                    raw_data_channel = raw_data.pick_channels([channel_name])
                elif fallback_channel in raw_data.info['ch_names']:
                    # 提取备用通道数据
                    raw_data_channel = raw_data.pick_channels([fallback_channel])
                    print(f"Warning: {channel_name} not found, using {fallback_channel} instead.")
                else:
                    raise ValueError(f"Neither {channel_name} nor {fallback_channel} found in {file_name}")

                # 提取数据为 NumPy 数组
                data = raw_data_channel.get_data()

                # 重新塑形为 (-1, frame_length) 的二维数组
                num_frames = data.shape[1] // frame_length  # 计算可以切割的帧数
                reshaped_data = data[:, :num_frames * frame_length].reshape(-1, frame_length)  # 重塑为二维数组

                # 删除最后 30 帧
                reshaped_data = reshaped_data[:-remove_last_frames, :]  # 删除最后 30 帧数据

                # 保存为 .npy 文件
                np.save(npy_path, reshaped_data)
                print(f"Converted {file_name} to {npy_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    input_folder = r"F:\dataset"  # 替换为你的 .rec 文件路径
    output_folder = "rec_data"  # 替换为你的目标 .npy 文件夹路径
    convert_edf_to_npy(input_folder, output_folder)

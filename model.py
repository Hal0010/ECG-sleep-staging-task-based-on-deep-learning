import torch
import torch.nn as nn


class CnnGruModel(nn.Module):
    """
    用于一维时间序列
    """
    def __init__(self, input_seq_len, hidden_dim, num_classes, conv_channels=[4, 8, 16], kernel_size=5,
                 dropout_prob=0.3):
        super(CnnGruModel, self).__init__()
        self.hidden_dim = hidden_dim

        # CNN部分，包含3个卷积层
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv_channels[0], kernel_size=kernel_size, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size,
                               stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.conv3 = nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=kernel_size,
                               stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout层

        # 计算经过CNN层后的序列长度
        conv_output_len = input_seq_len
        for _ in range(3):  # 三个卷积层
            conv_output_len = (conv_output_len - (kernel_size - 1) - 1) // 2 + 1  # 卷积层
            conv_output_len = (conv_output_len - 4) // 2 + 1  # 最大池化层

        if conv_output_len <= 0:
            raise ValueError("输入序列长度在经过卷积和池化后太短，请调整卷积核大小或输入长度。")

        self.output_seq_len = conv_output_len  # 输出的序列长度
        self.gru_input_size = conv_channels[2]  # GRU输入的维度

        # GRU部分
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=hidden_dim, batch_first=True)

        # 全连接部分
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # CNN部分
        x = x.unsqueeze(1)  # 添加通道维度，形状变为 (batch_size, 1, seq_len)
        x = self.dropout(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Dropout
        x = self.pool1(x)  # 最大池化
        x = self.dropout(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Dropout
        x = self.pool2(x)  # 最大池化
        x = self.dropout(torch.relu(self.conv3(x)))  # Conv3 + ReLU + Dropout
        x = self.pool3(x)  # 最大池化

        # 为GRU准备数据
        x = x.permute(0, 2, 1).contiguous()  # 变换形状为 (batch_size, seq_len, features)

        # GRU部分
        gru_out, _ = self.gru(x)  # GRU输出 (batch_size, seq_len, hidden_dim)

        # 在时间维度上聚合GRU的输出
        gru_agg = torch.mean(gru_out, dim=1)  # 对时间步进行平均池化 (batch_size, hidden_dim)

        # 全连接层用于分类
        out = self.fc(gru_agg)  # 使用聚合后的GRU输出进行分类

        return out


class CnnImageModel(nn.Module):
    """
    用于单通道图像
    """
    def __init__(self, input_channels, input_height, input_width, num_classes, conv_channels=[16, 32, 64],
                 kernel_size=3, dropout_prob=0.3):
        super(CnnImageModel, self).__init__()

        self.input_height = input_height
        self.input_width = input_width

        # CNN部分，使用nn.Sequential构建
        self.cnn = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积层
            nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积层
            nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算经过CNN层后的特征图大小
        self.fc_input_size = self._calculate_fc_input_size(input_height, input_width)

        # 全连接层用于分类
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input_size(self, height, width):
        # 计算经过每个卷积和池化层后的输出尺寸
        for _ in range(3):  # 三个卷积+池化层
            height = (height - 2) // 2 + 1  # 经过MaxPool2d(kernel_size=2, stride=2)后
            width = (width - 2) // 2 + 1

        # 返回全连接层的输入特征数量
        return 64 * height * width  # 64是最后一个卷积层的输出通道数

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度 (batch_size, 1, seq_len)

        x = self.cnn(x)  # 通过卷积层

        # 将输出展平为全连接层的输入
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, features)

        # 全连接层进行分类
        out = self.fc(x)
        return out


class CnnImageModelC2(nn.Module):
    """
    用于双通道图像数据
    """
    def __init__(self, input_channels, input_height, input_width, num_classes, conv_channels=[16, 32, 64],
                 kernel_size=3, dropout_prob=0.3):
        super(CnnImageModelC2, self).__init__()

        # CNN部分，使用nn.Sequential构建
        self.cnn = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积层
            nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积层
            nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2], kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算经过CNN层后的特征图大小
        self.fc_input_size = self._calculate_fc_input_size(input_height, input_width, len(conv_channels), kernel_size=2)

        # 全连接层用于分类
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input_size(self, height, width, num_pooling_layers, kernel_size):
        # 计算经过每个卷积和池化层后的输出尺寸
        for _ in range(num_pooling_layers):  # 三个卷积+池化层
            height = (height - kernel_size) // 2 + 1
            width = (width - kernel_size) // 2 + 1

        # 返回全连接层的输入特征数量
        return 64 * height * width  # 64是最后一个卷积层的输出通道数

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 改变维度为[batch_size, channels, height, width]
        # 输入形状：[batch_size, channels, height, width]
        x = self.cnn(x)

        # 将输出展平为全连接层的输入
        x = x.reshape(x.size(0), -1)  # 展平为 (batch_size, features)

        # 全连接层进行分类
        out = self.fc(x)
        return out


if __name__ == "__main__":
    # 创建一个CnnImageModelC2模型实例，输入为2通道，64x64的图像，5个类别
    model = CnnImageModelC2(input_channels=2, input_height=64, input_width=64, num_classes=5)
    # 模拟输入数据，形状为 [batch_size, height, width, channels]
    sample_input = torch.randn(128, 64, 64, 2)  # 128张64x64的图像，2个通道
    output = model(sample_input)

    print("Output shape:", output.shape)  # 输出的形状应为 [128, 5]，对应128个样本和5个分类

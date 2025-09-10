import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
import numpy as np
from model import CnnGruModel, CnnImageModel, CnnImageModelC2

from loaddata import SleepDataset
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, fold, save_path):
    """
    绘制并保存混淆矩阵图。
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Fold {fold}')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 在每个格子中标注值
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def train(model, train_loader, criterion, optimizer, device):
    """
    模型训练部分。
    """
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    return avg_train_loss


def val(model, val_loader, criterion, device):
    """
    模型验证部分。
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions)

    return avg_val_loss, accuracy, f1, conf_matrix, kappa


def main(data_folder_path, label_folder_path, batch_size, num_epochs, device, weight_path=None):
    """
    主函数：进行五折交叉验证。
    """
    # 创建tensorboard日志目录
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dataset = SleepDataset(data_folder_path, label_folder_path, 'merged_sleep_data_combined.npz')
    total_samples = len(dataset)

    kf = KFold(n_splits=5, shuffle=True)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(np.arange(total_samples))):
        print(f"\nFold {fold + 1}/{kf.n_splits}")

        # 为每个fold创建一个writer
        writer = SummaryWriter(f'{log_dir}/fold_{fold + 1}')

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)  # 创建dataloader
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

        model = CnnImageModelC2(2, 64, 64, 5)  # 实例化模型
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            avg_train_loss = train(model, train_loader, criterion, optimizer, device)
            avg_val_loss, accuracy, f1, conf_matrix, kappa = val(model, val_loader, criterion, device)

            # 记录到tensorboard
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Metrics/accuracy', accuracy, epoch)
            writer.add_scalar('Metrics/f1_score', f1, epoch)
            writer.add_scalar('Metrics/kappa', kappa, epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Kappa: {kappa:.4f}")

        # 保存当前fold的模型权重
        model_save_path = os.path.join(log_dir, f"fold_{fold + 1}_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model weights for fold {fold + 1} at {model_save_path}")

        # 绘制并保存最终验证集混淆矩阵
        classes = [f'Class {i}' for i in range(conf_matrix.shape[0])]
        cm_save_path = os.path.join(log_dir, f'fold_{fold + 1}_conf_matrix.png')
        plot_confusion_matrix(conf_matrix, classes, fold + 1, cm_save_path)

        print(f"Confusion Matrix (Fold {fold + 1}):\n{conf_matrix}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'conf_matrix': conf_matrix,
            'kappa': kappa
        })

        # 关闭当前fold的writer
        writer.close()

    print("\nCross-validation Results:")
    for result in fold_results:
        print(f"Fold {result['fold']} - "
              f"Accuracy: {result['accuracy']:.4f}, F1 Score: {result['f1_score']:.4f}, Kappa: {result['kappa']:.4f}")
        print(f"Confusion Matrix (Fold {result['fold']}):\n{result['conf_matrix']}")

    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])
    avg_kappa = np.mean([r['kappa'] for r in fold_results])
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Kappa Score: {avg_kappa:.4f}")


if __name__ == "__main__":
    data_folder_path = "data_reading/rec_data"  # .npy文件文件夹
    label_folder_path = "data_reading/label"  # 标签文件文件夹
    batch_size = 64
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(data_folder_path, label_folder_path, batch_size, num_epochs, device)

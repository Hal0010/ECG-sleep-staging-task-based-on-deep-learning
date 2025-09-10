import os


def remove_empty_lines_from_labels(label_folder, remove_last_lines=30):
    # 遍历标签文件夹中的所有txt文件
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)

        # 检查文件是否是txt文件
        if label_file.endswith('.txt'):
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # 删除空行并去除每行的前后空白
            cleaned_lines = [line.strip() for line in lines if line.strip()]

            # 删除最后30行标签数据
            cleaned_lines = cleaned_lines[:-remove_last_lines]  # 删除最后30行

            # 将清理后的内容重新写入文件
            with open(label_path, 'w') as file:
                file.writelines("\n".join(cleaned_lines) + "\n")

            print(f"Empty lines removed and last 30 lines deleted from {label_file}")
        else:
            print(f"Skipping non-txt file: {label_file}")


if __name__ == "__main__":
    label_folder = "label"  # 标签文件的文件夹路径
    remove_empty_lines_from_labels(label_folder)

import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        # 初始化变量
        self.image_folder = None
        self.image_list = []
        self.current_index = 0
        self.labeled_index = -1  # 已标注数据的上限
        self.labels = {}

        # 设置界面布局
        self.image_label = QLabel("No Image Loaded", self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton("Load Image Folder", self)
        self.load_button.clicked.connect(self.load_images)

        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_image)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_image)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_images(self):
        """加载图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.image_folder = folder
        self.image_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_list.sort()  # 可选：排序图片列表
        self.current_index = 0
        self.labeled_index = -1
        self.labels = {}
        if self.image_list:
            self.display_image()

    def display_image(self):
        """显示当前图片"""
        if 0 <= self.current_index < len(self.image_list):
            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.image_label.setText("Failed to load image.")
            else:
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("No more images.")

    def keyPressEvent(self, event):
        """处理键盘事件"""
        if not self.image_list:
            return

        if event.key() == Qt.Key_L:  # 'L' 键标注为 1
            self.labels[self.image_list[self.current_index]] = "1"
            self.labeled_index = max(self.labeled_index, self.current_index)  # 更新已标注数据的上限
            self.save_labels()  # 实时保存
            self.current_index += 1
            self.display_image()
        elif event.key() == Qt.Key_K:  # 'K' 键标注为 0
            self.labels[self.image_list[self.current_index]] = "0"
            self.labeled_index = max(self.labeled_index, self.current_index)  # 更新已标注数据的上限
            self.save_labels()  # 实时保存
            self.current_index += 1
            self.display_image()
        elif event.key() == Qt.Key_Up:  # 上方向键（往前翻）
            self.prev_image()
        elif event.key() == Qt.Key_Down:  # 下方向键（往后翻）
            self.next_image()

        if self.current_index >= len(self.image_list):
            self.image_label.setText("Labeling completed!")
            self.save_labels()

    def prev_image(self):
        """往前翻阅图片"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()
        else:
            self.image_label.setText("Already at the first image.")

    def next_image(self):
        """往后翻阅图片（限制为已标注数据范围）"""
        if self.current_index < self.labeled_index + 1:
            self.current_index += 1
            self.display_image()
        else:
            self.image_label.setText("Cannot go beyond labeled images.")

    def save_labels(self):
        """保存标注结果到每张图片对应的同名 .txt 文件"""
        if self.image_folder and self.labels:
            for image, label in self.labels.items():
                # 生成 .txt 文件路径
                txt_filename = os.path.splitext(image)[0] + ".txt"
                txt_path = os.path.join(self.image_folder, txt_filename)
                
                # 写入标注内容
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(label)  # 仅写入标注值（"0" 或 "1"）


if __name__ == "__main__":
    app = QApplication([])
    window = ImageLabelingApp()
    window.show()
    app.exec_()

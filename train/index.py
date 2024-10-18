from ultralytics import YOLO
import torch

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/11/yolo11.yaml"

# 预训练模型
pre_model_name = './pretrain/yolo11n.pt'


# 数据集配置文件
data_yaml_path = 'ultralytics/cfg/datasets/steel-plate.yaml'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

if __name__ == '__main__':
    model = YOLO(model_yaml_path).load(pre_model_name)
    results = model.train(data=data_yaml_path, epochs=100, batch=4, name='train_v11', workers=8, patience=50)

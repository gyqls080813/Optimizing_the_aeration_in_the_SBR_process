import os
import pandas as pd
import torch
import torch.nn as nn

class OnlineMLPRegressor:
    def __init__(self, input_size=4, hidden_size=64, output_size=1, learning_rate=0.001, model_path=None, max_epochs=10000, min_val=0.0, max_val=1000.0):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.max_epochs = max_epochs

        # 스케일링을 위한 최소/최대값 설정
        self.min_val = min_val
        self.max_val = max_val

        # 모델 경로가 지정되어 있고, 해당 파일이 존재할 경우 모델을 불러옴
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found, initializing with random weights.")
            self.initialize_weights()

    def initialize_weights(self):
        # 모델의 가중치를 초기화
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def scale_data(self, X):
        # 데이터를 0에서 1 사이로 정규화
        return (X - self.min_val) / (self.max_val - self.min_val)

    def fit(self, X, y):
        self.model.train()
        # 데이터를 스케일링
        X = self.scale_data(X)
        X, y = X.to(self.model[0].weight.device), y.to(self.model[0].weight.device)

        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss = self.criterion(predictions.squeeze(), y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")
        return loss.item()

    def predict(self, X):
        self.model.eval()
        # 데이터를 스케일링
        X = self.scale_data(X)
        X = X.to(self.model[0].weight.device)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu()

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"Model loaded from {file_path}")
        else:
            print("Model file not found, initializing with random weights.")
            self.initialize_weights()

def load_and_preprocess_data(root_dir, sensor_columns, start_time=3300, end_time=3600):
    data_list = []

    for label in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label)
        if os.path.isdir(folder_path):
            concentration = float(label.replace('ppm', '').strip())
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                start_time_value = pd.to_datetime(df['Timestamp'].iloc[0], format='%Y-%m-%d %H:%M:%S')
                df['Time_in_seconds'] = (pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S') - start_time_value).dt.total_seconds()

                df_filtered = df[(df['Time_in_seconds'] >= start_time) & (df['Time_in_seconds'] <= end_time)].copy()

                if len(df_filtered) > 0:
                    df_filtered['Concentration'] = concentration
                    df_filtered = df_filtered[sensor_columns + ['Concentration']]
                    data_list.append(df_filtered)

    if data_list:
        data = pd.concat(data_list, ignore_index=True)
        data[sensor_columns] = data[sensor_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # 데이터를 텐서로 변환하여 반환
        X = torch.tensor(data[sensor_columns].values, dtype=torch.float32)
        y = torch.tensor(data['Concentration'].values, dtype=torch.float32)
        return X, y
    else:
        return torch.empty(0), torch.empty(0)

def train_and_evaluate_model(root_dir, sensor_columns, model_save_path):
    # 데이터 로드 및 전처리
    X, y = load_and_preprocess_data(root_dir, sensor_columns)
    if X.size(0) == 0:
        print("No data available.")
        return
    regression_model_path = "/path/to/your/directory/"
    regression_model = OnlineMLPRegressor(model_path=regression_model_path, max_epochs=10000)  # 에폭 설정

    # 데이터 학습
    regression_model.fit(X, y)

    # 모델 저장
    regression_model.save_model(model_save_path)

# 사용 예제
if __name__ == "__main__":
    root_dir = "/path/to/your/directory/"
    sensor_columns = ['Clearlight', 'Red', 'Green', 'Blue']
    model_save_path = "/path/to/your/directory/"
    train_and_evaluate_model(root_dir, sensor_columns, model_save_path)

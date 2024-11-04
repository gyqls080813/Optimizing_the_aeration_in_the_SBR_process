import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from collections import deque
import os

class DQNLSTMModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNLSTMModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.4
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.low_reward_threshold = -1  # 탐색 비율을 늘리기 위한 보상 기준
        self.high_reward_threshold = 1  # 탐색 비율을 줄이기 위한 보상 기준


        # LSTM layer definition (input_size=4, hidden_size=32)
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(32, 24)
        self.fc2 = nn.Linear(24, 900)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def adjust_exploration(self, reward):
        if reward < self.low_reward_threshold:
            # 낮은 보상이 나오면 탐색 비율을 증가시키고 감쇠율을 1로 설정
            self.epsilon = min(1.0, self.epsilon + 0.1)
            self.epsilon_decay = 1.0
        elif reward > self.high_reward_threshold:
            # 높은 보상이 나오면 탐색 비율을 줄이고 감쇠율을 조정하여 수렴을 유도
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995

        # 안정된 탐색을 위해 일정 에피소드 동안 조정 유지
        self.stable_count = 10


    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        # Output scaled to range [0, 900]
        x = torch.sigmoid(x)
        return x

    def memorize(self, state, action, reward):
        self.memory.append((state, action, reward))
        

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: randomly choose a value between 0 and 900
            return np.random.uniform(0, 900)
        
        self.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)  # 배치 차원 추가
            act_values = self(state).cpu().numpy()  # Q-값을 가져옴
        self.train()
        
        return np.argmax(act_values[0])  # 최대 Q-값의 인덱스를 반환

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward in minibatch:
            # Set target as reward
            target = reward
            
            # Add batch dimension
            state = state.unsqueeze(0)
            
            # Calculate predicted Q-values
            predicted = self(state).clone()
            
            # Update target value for the selected action
            predicted[0, action] = target
            
            # Store states and updated targets
            states.append(state)
            targets_f.append(predicted)

        # Train the model
        states = torch.cat(states)
        targets_f = torch.cat(targets_f)
        
        # Zero the gradients
        self.optimizer.zero_grad()
        # Calculate loss
        loss = self.loss_fn(self(states), targets_f)
        # Backpropagation
        loss.backward()
        # Update weights
        self.optimizer.step()

        # Reduce epsilon to decrease exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def load(self, name):
        self.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.state_dict(), name)

class OnlineMLPRegressor:
    def __init__(self, input_size=4, hidden_size=64, output_size=1, learning_rate=0.001, model_path=None, max_epochs=100, min_val=0.0, max_val=1000.0):
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

    def strong_fit(self, X_sensor, y_real, loss_threshold, max_extra_epochs):
        """강한 학습: 센서 데이터를 기반으로 예측하고, 실제 농도와의 손실을 최소화"""
        self.model.train()

        # 센서 데이터를 스케일링
        X_sensor_scaled = self.scale_data(X_sensor).to(self.model[0].weight.device)
        X_sensor_scaled.requires_grad_(True)  # 역전파를 위한 설정

        if not isinstance(y_real, torch.Tensor):
            y_real = torch.tensor(y_real, dtype=torch.float32)

        y_real = y_real.to(self.model[0].weight.device)

        epoch_count = 0
        while True:
            self.optimizer.zero_grad()
            predictions = self.model(X_sensor_scaled)
            y_real_expanded = y_real.expand_as(predictions)  # 예측 크기에 맞춰 y_real 확장
            loss = self.criterion(predictions.squeeze(), y_real_expanded.squeeze())
            loss.backward()  # 역전파

            # 가중치 업데이트
            self.optimizer.step()

            epoch_count += 1

            # 손실이 임계값 이하로 떨어지면 중지
            if loss.item() < loss_threshold:
                print(f"Stopping early at epoch {epoch_count} with loss: {loss.item():.6f}")
                break

            # 최대 에포크 초과 시 중지
            if epoch_count >= max_extra_epochs:
                print(f"Reached max extra epochs with loss: {loss.item():.6f}")
                break

        # 예측 농도를 반환
        return predictions.cpu().detach(), loss.item()

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

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f"Model loaded from {file_path}")
        else:
            print("Model file not found, initializing with random weights.")
            self.initialize_weights()

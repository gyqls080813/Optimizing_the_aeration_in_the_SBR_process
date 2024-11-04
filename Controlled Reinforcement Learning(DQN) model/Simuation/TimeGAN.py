import os
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(root_dir, sensor_columns, start_time=600, end_time=3400, min_val=0.0, max_val=1000.0, min_time=0.0, max_time=3600.0):
    def normalize_sensor_values(values, min_val, max_val):
        return 2 * ((values - min_val) / (max_val - min_val)) - 1

    data_list = []
    
    for label in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label)
        if os.path.isdir(folder_path):
            concentration = float(label.replace('ppm', '').strip())
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                
                start_time_value = datetime.strptime(df['Timestamp'].iloc[0], '%Y-%m-%d %H:%M:%S')
                df['Time_in_seconds'] = df['Timestamp'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - start_time_value).total_seconds())
                
                df_filtered = df[(df['Time_in_seconds'] >= start_time) & (df['Time_in_seconds'] <= end_time)].copy()
                
                if len(df_filtered) > 0:
                    df_filtered['Concentration'] = concentration
                    data_list.append(df_filtered)
    
    if data_list:
        data = pd.concat(data_list, ignore_index=True)
        data[sensor_columns] = data[sensor_columns].apply(lambda x: normalize_sensor_values(x, min_val, max_val))
        return data
    else:
        return pd.DataFrame()

# 펌프 상태별로 데이터를 필터링하고 시간 구간에 맞게 정규화하는 함수
def filter_and_normalize_data_by_pump_status(data, pump_status, min_time, max_time):
    filtered_data = data[data[['Mixer', 'Input_pump', 'Carbon_pump', 'Air_pump', 'Output_pump', 'Sample_pump', 'Solution1_pump', 'Solution2_pump']].apply(
        lambda row: (row.values == pump_status).all(), axis=1)].copy()  # .copy() 메소드 추가
    
    # 시간 값을 상태에 맞춰 정규화
    filtered_data['Time_in_seconds'] = 2 * ((filtered_data['Time_in_seconds'] - min_time) / (max_time - min_time)) - 1
    
    return filtered_data

# Conditional Generator 정의 (조건 추가)
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        input_data = torch.cat((noise, condition), dim=1)
        return self.model(input_data)

# Conditional Discriminator 정의 (조건 추가)
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, data, condition):
        input_data = torch.cat((data, condition), dim=1) 
        return self.model(input_data)

# 하이퍼파라미터 설정 함수
def get_hyperparameters():
    hyperparameters = {
        'noise_dim': 100,
        'sensor_dim': 4,
        'pump_status_dim': 8,
        'condition_dim': 10,
        'num_epochs': 3000,
        'batch_size': 64,
        'learning_rate_G': 0.0003,
        'learning_rate_D': 0.0001,
        'save_interval': 50
    }
    return hyperparameters

# TimeGAN 학습 파이프라인 함수 (모델 저장 및 상태별 손실 기록 추가)
def train_timegan_pipeline_sequential(data, device):
    # 펌프 상태 정의
    pump_statuses = {
        'Status3': ([0, 0, 0, 0, 0, 0, 0, 0], 3000, 3300)
    }

    # 각 상태에 맞는 데이터 필터링 및 정규화, DataLoader 준비
    dataloaders = {}
    for status_name, (pump_status, min_time, max_time) in pump_statuses.items():
        filtered_data = filter_and_normalize_data_by_pump_status(data, pump_status, min_time, max_time)
        if filtered_data.empty:
            print(f"No data found for {status_name}. Skipping...")
            continue
        
        # 텐서로 변환
        sensor_values = torch.tensor(filtered_data[['Clearlight', 'Red', 'Green', 'Blue']].values, dtype=torch.float32)
        pump_status = torch.tensor(filtered_data[['Mixer', 'Input_pump', 'Carbon_pump', 'Air_pump', 'Output_pump', 'Sample_pump', 'Solution1_pump', 'Solution2_pump']].values, dtype=torch.float32)
        
        # 농도를 음수로 변환하여 입력
        concentration = torch.tensor(-filtered_data['Concentration'].values, dtype=torch.float32).unsqueeze(1)  # 음수로 변환된 농도
        
        time_in_seconds = torch.tensor(filtered_data['Time_in_seconds'].values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(sensor_values, pump_status, concentration, time_in_seconds)
        dataloaders[status_name] = DataLoader(dataset, batch_size=32, shuffle=True)

    # 하이퍼파라미터 불러오기
    hyperparams = get_hyperparameters()
    noise_dim = hyperparams['noise_dim']
    sensor_dim = hyperparams['sensor_dim']
    condition_dim = hyperparams['condition_dim']
    num_epochs = hyperparams['num_epochs']
    save_interval = hyperparams['save_interval']

    generator = ConditionalGenerator(noise_dim=noise_dim, condition_dim=condition_dim, output_dim=sensor_dim).to(device)
    discriminator = ConditionalDiscriminator(input_dim=sensor_dim, condition_dim=condition_dim).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=hyperparams['learning_rate_G'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hyperparams['learning_rate_D'])
    criterion = nn.BCELoss()

    # 손실 기록용 딕셔너리
    loss_history = {status: [] for status in pump_statuses.keys()}

    for epoch in range(1, num_epochs + 1):
        for status_name, dataloader in dataloaders.items():
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            print(f"Epoch {epoch} - Training for {status_name}...")

            for sensor_values, pump_status, concentration, time_in_seconds in dataloader:
                batch_size = sensor_values.size(0)

                sensor_values = sensor_values.to(device)
                pump_status = pump_status.to(device)
                concentration = concentration.to(device)  # 음수로 변환된 농도 값
                time_in_seconds = time_in_seconds.to(device)

                noise = torch.randn(batch_size, noise_dim, device=device)

                condition = torch.cat((pump_status, concentration, time_in_seconds), dim=1)

                optimizer_D.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                real_loss = criterion(discriminator(sensor_values, condition), real_labels)
                fake_sensor_values = generator(noise, condition)
                fake_loss = criterion(discriminator(fake_sensor_values.detach(), condition), fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(fake_sensor_values, condition), real_labels)
                g_loss.backward()
                optimizer_G.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()

            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            loss_history[status_name].append([epoch, avg_d_loss, avg_g_loss])

            print(f"Epoch [{epoch}/{num_epochs}] for {status_name}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}")

              # 모델 저장
            if epoch % save_interval == 0 or epoch == num_epochs:
                os.makedirs(f"./model", exist_ok=True)  # model 폴더에 저장
                torch.save(generator.state_dict(), f"./model/timegan/generator_{status_name}_epoch_{epoch}.pth")
                torch.save(discriminator.state_dict(), f"./model/timegan/discriminator_{status_name}_epoch_{epoch}.pth")
                print(f"Model saved for {status_name} at epoch {epoch}")

                # 손실 기록을 상태별로 CSV 파일에 저장
                os.makedirs(f"./loss", exist_ok=True)  # loss 폴더에 저장
                loss_df = pd.DataFrame(loss_history[status_name], columns=["Epoch", "D_Loss", "G_Loss"])
                loss_df.to_csv(f"./loss/losses_{status_name}.csv", index=False)
                print(f"Losses saved for {status_name} at epoch {epoch}")

# 데이터 로드 및 학습 파이프라인 실행
root_dir = "/path/to/your/directory/"
sensor_columns = ['Clearlight', 'Red', 'Green', 'Blue']

# 전처리된 데이터 로드
processed_data = load_and_preprocess_data(root_dir, sensor_columns)

# GPU 디바이스 확인 및 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 펌프 상태별로 학습 실행
train_timegan_pipeline_sequential(processed_data, device)
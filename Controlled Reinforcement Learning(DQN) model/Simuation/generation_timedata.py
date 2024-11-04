import torch
import numpy as np
import torch.nn as nn
import pandas as pd

# Conditional Generator 정의 (데이터 생성용으로 이미 학습된 모델과 동일한 구조)
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
            nn.Tanh()  # 정규화된 데이터에 맞게 Tanh 활성화 함수 사용
        )

    def forward(self, noise, condition):
        input_data = torch.cat((noise, condition), dim=1)
        return self.model(input_data)

# 역 스케일링 함수
def inverse_scaling(value, min_val, max_val):
    return ((value + 1) / 2) * (max_val - min_val) + min_val

# 수정된 generate_data 함수
def generate_data(model_path, time_range, concentration, noise_dim=100, sensor_dim=4, device='cpu', min_val=0.0, max_val=1000.0):
    generated_data = []

    # time_range는 튜플이므로 start_time과 end_time을 추출
    start_time, end_time = time_range

    # 시간 범위 생성
    time_values = np.linspace(start_time, end_time, end_time - start_time)

    # 모델 불러오기
    generator = ConditionalGenerator(noise_dim=noise_dim, condition_dim=2, output_dim=sensor_dim).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()  # 평가 모드로 전환

    for time_val in time_values:
        # 노이즈 생성
        noise = torch.randn(1, noise_dim, device=device)

        # 조건 결합 (음수 농도 + 정규화된 시간)
        normalized_time = (time_val - start_time) / (end_time - start_time) * 2 - 1
        condition = torch.tensor([[-concentration, normalized_time]], dtype=torch.float32, device=device)

        # 데이터 생성
        with torch.no_grad():
            generated_sample = generator(noise, condition).cpu().numpy()

        # 역 스케일링 적용
        clearlight = inverse_scaling(generated_sample[0][0], min_val, max_val)
        red = inverse_scaling(generated_sample[0][1], min_val, max_val)
        green = inverse_scaling(generated_sample[0][2], min_val, max_val)
        blue = inverse_scaling(generated_sample[0][3], min_val, max_val)

        # 결과 저장
        generated_data.append({
            'Time': time_val,
            'Clearlight': clearlight,
            'Red': red,
            'Green': green,
            'Blue': blue,
        })

    return pd.DataFrame(generated_data)
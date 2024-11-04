import torch
import random
import numpy as np
import pandas as pd
from SBR_Environment import SBR_Environment
from main_model import DQNLSTMModel, OnlineMLPRegressor
from generation_timedata import generate_data
import torch
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_model_path = "/path/to/your/directory/"
timegan_model_path = "/path/to/your/directory/"
save_dpn = "/path/to/your/directory/"
save_dqn_result = "/path/to/your/directory/"

# 모델을 메모리에 먼저 로드
def load_model(model_path, device):
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())  # 파일을 메모리에 로드
    state_dict = torch.load(buffer, map_location=device)  # 메모리에서 state_dict 불러오기
    return state_dict

def calculate_reward(predicted_concentration, target_concentration=1.0, lower_bound=0.5, upper_bound=1.5, penalty_scale=2.0, reward_scale=10.0):
    if lower_bound <= predicted_concentration <= upper_bound:
        # 목표값에 더 가까울수록 더 높은 보상을 주는 방식으로 수정
        reward = reward_scale * (1.0 - abs(target_concentration - predicted_concentration))
    else:
        # 목표 범위를 벗어났을 때 페널티 부여
        distance = abs(target_concentration - predicted_concentration)
        reward = -penalty_scale * distance ** 2
    return reward


def main():
    num_step = 10000
    batch_size = 32
    agent = DQNLSTMModel(4, 900).to(device)

    # 회귀 모델 초기화 및 저장 경로 설정
    mlp_model_path = mlp_model_path
    regression_model = OnlineMLPRegressor(model_path=mlp_model_path)  # 저장된 모델 불러오기

    # main 함수에서 호출 예시
    model_path = timegan_model_path

    # CSV 파일에 저장할 데이터 초기화
    results_data = []

    # 초기 사이클 농도 계산
    env = SBR_Environment()
    first_concentration, reaction_concentration_1 = env.step(0,900)  # 첫 번째 시작 농도가 없으므로 0으로 시작  # 첫 번째 사이클: 폭기 시간 900초 고정
    second_concentration, reaction_concentration_2 = env.step(first_concentration,900)  # 두 번째 사이클  # 두 번째 사이클: 폭기 시간 900초 고정

    # 첫 번째 및 두 번째 사이클 결과 저장
    results_data.append({
        'Start concentration': env.start_concentration,
        'Episode': 1,
        'Aeration_Time': 900,
        'Post_Aeration_Concentration': first_concentration
    })
    print(f"Data appended for Episode 1: input concentration: {env.start_concentration}, Start concentration: {reaction_concentration_1}, Aeration_Time: 900, Post_Aeration_Concentration: {first_concentration}")

    results_data.append({
        'Start concentration': env.start_concentration,
        'Episode': 2,
        'Aeration_Time': 900,
        'Post_Aeration_Concentration': second_concentration
    })
    print(f"Data appended for Episode 2: Start concentration: {env.start_concentration}, , Start concentration: {reaction_concentration_2}, Aeration_Time: 900, Post_Aeration_Concentration: {second_concentration}")

    for episode in range(3, num_step + 1):
        weight_factor = 0.99**(episode - 2) 
        if episode % 50 == 0:
            env.start_concentration = random.randint(35, 40)
            print(f"Start concentration updated to: {env.start_concentration}")
        
        if episode % 2000 == 0:
            # 에피소드에 따라 순서대로 공식 업데이트
            formula_index = (episode // 2000) % len(env.formulas)
            env.current_formula = env.formulas[formula_index]
            print(f"Current formula updated to formula index: {formula_index}")

        if episode == 3:
            previous_concentration = second_concentration
            previous_sensor_data = generate_data(
            model_path=model_path,
            time_range=(3000, 3300),
            concentration=previous_concentration,
            device='cpu'
        )
            current_concentration, reaction_concentration = env.step(previous_concentration,900)  # 두 번째 반복의 농도를 사용하여 갱신
        else:
            current_concentration, reaction_concentration = env.step(previous_concentration, previous_action)  # 이전 사이클의 농도를 사용하여 갱신
    
        if previous_sensor_data is not None and 'Concentration' in previous_sensor_data.columns:
            concentration = previous_sensor_data['Concentration'].iloc[0]  # 이전 농도 값 사용
            # 센서 데이터를 텐서로 변환
            sensor_data = torch.tensor(previous_sensor_data[['Clearlight', 'Red', 'Green', 'Blue']].values, dtype=torch.float32)

            if sensor_data.ndim == 2 and sensor_data.size(0) > 0:  # 데이터가 2차원 배열이고, 비어 있지 않은 경우만 업데이트
                # 최대 10개의 데이터만 선택
                num_samples = min(10, sensor_data.size(0))
                selected_sensor_data = sensor_data[:num_samples]
                # 목표값을 선택한 센서 데이터 개수에 맞게 확장
                target = torch.full((num_samples,), concentration, dtype=torch.float32)

                # 모델 업데이트 및 손실 계산
                loss = regression_model.fit(selected_sensor_data, target)  # 모델 업데이트
                regression_model.save_model(mlp_model_path)
                
                # 손실 값 출력
                if loss is not None:
                    print(f"Regression model updated with loss: {loss:.6f}")
                else:
                    print("Loss could not be calculated.")


        # 센서 데이터 생성
        current_sensor_data = generate_data(
            model_path=model_path,
            time_range=(3000, 3300),
            concentration=current_concentration,
            device='cpu'
        )
        current_sensor_data[['Clearlight', 'Red', 'Green', 'Blue']] *= weight_factor

        # Reward calculation and PPO model updates
        # Update the SAC algorithm
        if episode >= 3:
            # Convert sensor data to tensor and predict concentrations
            previous_sensor_data_tensor = torch.tensor(
                previous_sensor_data[['Clearlight', 'Red', 'Green', 'Blue']].values, dtype=torch.float32).to(device)
            current_sensor_data_tensor = torch.tensor(
                current_sensor_data[['Clearlight', 'Red', 'Green', 'Blue']].values, dtype=torch.float32).to(device)

            # Perform strong fitting using the regression model
            predicted_concentrations, strong_loss = regression_model.strong_fit(
                current_sensor_data_tensor, current_concentration, loss_threshold=0.001, max_extra_epochs=1000)
            regression_model.save_model(mlp_model_path)
            print(f"Strong fitting complete. Final loss: {strong_loss:.6f}")

            actions = []
            with torch.no_grad():
                for i in range(previous_sensor_data_tensor.shape[0]):  # 300번 반복
                    action = agent.act(previous_sensor_data_tensor[i].unsqueeze(0))  # (1, 4) 형태로 변환
                    actions.append(action)
            selected_action = np.bincount(actions).argmax()

            # Calculate rewards based on the predicted concentrations
            rewards = torch.tensor([
                calculate_reward(pred.item(), target_concentration=1.0, lower_bound=0.8, upper_bound=1.2, penalty_scale=0.5)
                for pred in predicted_concentrations
            ])

            # Total reward calculation
            total_reward = rewards.mean().item()
            # 탐색 비율 조정
            agent.adjust_exploration(total_reward)
            agent.memorize(previous_sensor_data_tensor, selected_action, total_reward)
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)

            # Record results for each episode
            results_data.append({
                'Input concentration': env.start_concentration,
                'Start concentration' : reaction_concentration,
                'Episode': episode,
                'Aeration_Time': selected_action,
                'Reward' : total_reward,
                'Average_predict_concentration' : predicted_concentrations.mean(),
                'Post_Aeration_Concentration': current_concentration
            })
            print(f"Data appended for Episode {episode}: input concentration: {env.start_concentration}, Start concentration: {reaction_concentration}, Aeration_Time: {selected_action}, Reward : {total_reward}, Average_predict_concentration : {predicted_concentrations.mean()}, Post_Aeration_Concentration': {current_concentration}")

        # 농도 데이터를 current_sensor_data에 추가할 조건 (회귀 모델 업데이트용)
        if episode % 10 == 0:
            current_sensor_data['Concentration'] = current_concentration
            agent.save(save_dpn)

        # 현재 사이클의 센서 데이터를 다음 사이클에 활용
        previous_concentration, previous_sensor_data = current_concentration, current_sensor_data
        previous_action = selected_action 

    # 결과 데이터를 CSV 파일로 저장
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(save_dqn_result, index=False)


if __name__ == "__main__":
    main()

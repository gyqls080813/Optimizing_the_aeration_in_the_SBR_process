import numpy as np
import random

class SBR_Environment:
    def __init__(self, aeration_time=None, total_episodes=10000):
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.start_concentration = random.randint(20, 50)
        self.aeration_time = aeration_time  # 초기 폭기 시간 설정 (옵션)
        self.formulas = [
            lambda x, y: y * (1 - np.sqrt(0.001 * x)),
            lambda x, y: y / (1 + 0.003 * x),
            lambda x, y: y * np.exp(-0.00197 * (x * 1 / (1 + np.exp(-x)))),
            lambda x, y: y * np.exp(-0.002 * x),
            lambda x, y: y * (1 - np.tanh(0.00151 * x))
        ]
        self.current_formula = self.formulas[0]

    def reset(self):
        """매 에피소드마다 환경 초기화"""
        self.current_episode += 1
        print(f"Current episode: {self.current_episode}")  # 디버깅을 위한 출력
        if self.current_episode % 50 == 0:
            self.start_concentration = random.randint(35, 40)
            print(f"Start concentration updated to: {self.start_concentration}")  # 디버깅을 위한 출력
        if self.current_episode % 2000 == 0:
            # 현재 에피소드 수에 따라 순서대로 공식 선택
            formula_index = (self.current_episode // 2000) % len(self.formulas)
            self.current_formula = self.formulas[formula_index]
            print(f"Current formula updated to formula index: {formula_index}")  # 디버깅을 위한 출력
        return self.start_concentration

    def step(self, current_concentration, aeration_time=None):
        """
        주어진 농도를 가지고 다음 상태로 이동
        - current_concentration: 현재 폭기 후 농도
        - aeration_time: 폭기 시간 (옵션, None일 경우 기본값 사용)
        """
        # 폭기 시간을 매개변수로 받아서 설정
        if aeration_time is not None:
            self.aeration_time = aeration_time
        
        # 반응 전 농도 계산
        reaction_concentration = (1/24 * self.start_concentration) + (23/24 * current_concentration)
        
        # 폭기 후 농도 계산 (현재 적용 중인 식 사용)
        post_aeration_concentration = self.current_formula(self.aeration_time, reaction_concentration)
        
        return post_aeration_concentration, reaction_concentration

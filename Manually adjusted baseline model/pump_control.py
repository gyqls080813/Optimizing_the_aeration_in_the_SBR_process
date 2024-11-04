import RPi.GPIO as GPIO
import shared_data
import time
import os
import aeration_adjust  # 폭기 시간 조정을 위한 모듈
import sensor_logging

# GPIO 핀 설정
def setup_pins():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(24, GPIO.OUT)  # Mixer
    GPIO.setup(27, GPIO.OUT)  # Input_pump
    GPIO.setup(18, GPIO.OUT)  # Carbon_pump
    GPIO.setup(23, GPIO.OUT)  # Air_pump
    GPIO.setup(22, GPIO.OUT)  # Output_pump
    GPIO.setup(14, GPIO.OUT)  # Sample_pump
    GPIO.setup(15, GPIO.OUT)  # Solution1_pump
    GPIO.setup(25, GPIO.OUT)  # Solution2_pump

# 펌프 상태를 지속적으로 업데이트하는 함수
def update_pump_status_continuous():
    while not shared_data.stop_thread:
        shared_data.data_store['Mixer'] = GPIO.input(24) == GPIO.HIGH
        shared_data.data_store['Input_pump'] = GPIO.input(27) == GPIO.HIGH
        shared_data.data_store['Carbon_pump'] = GPIO.input(18) == GPIO.HIGH
        shared_data.data_store['Air_pump'] = GPIO.input(23) == GPIO.HIGH
        shared_data.data_store['Output_pump'] = GPIO.input(22) == GPIO.HIGH
        shared_data.data_store['Sample_pump'] = GPIO.input(14) == GPIO.HIGH
        shared_data.data_store['Solution1_pump'] = GPIO.input(15) == GPIO.HIGH
        shared_data.data_store['Solution2_pump'] = GPIO.input(25) == GPIO.HIGH
        time.sleep(1)

# 첫 번째 사이클인지 확인하는 함수
def is_first_cycle():
    csv_file = "/home/pi/wwtp/sensor_concentration/average_sensor_data.csv"
    if not os.path.exists(csv_file):
        return True  # CSV 파일이 없으면 첫 번째 사이클로 간주
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        if len(lines) <= 1:  # 첫 번째 헤더만 존재하거나 데이터가 없으면 첫 번째 사이클
            return True
    return False

def adjust_aeration_time():
    if is_first_cycle():
        # 첫 번째 사이클인 경우 고정된 폭기 시간 설정
        print("첫 번째 사이클입니다. 폭기 시간을 900초로 설정합니다.")
        aeration_time = 900
    else:
        # 첫 번째 사이클이 아닌 경우 이전 사이클 데이터를 기반으로 폭기 시간 조정
        previous_data = aeration_adjust.get_previous_cycle_data()
        if previous_data:
            aeration_time = aeration_adjust.adjust_aeration_time(previous_data)
        else:
            aeration_time = 900  # 이전 데이터가 없는 경우 기본 900초 사용
    
    print(f"설정된 폭기 시간: {aeration_time}초")
    return aeration_time

# 펌프 제어 작업 처리
def run_pump_sequence():
    print('교반 시작')
    GPIO.output(24, True)  # Mixer 시작
    shared_data.data_store['Mixer'] = True

    print('유입 펌프 시작')
    GPIO.output(27, True)  # Input_pump 시작
    shared_data.data_store['Input_pump'] = True
    
    print('탄소 펌프 시작')
    GPIO.output(18, True)  # Carbon_pump 시작
    shared_data.data_store['Carbon_pump'] = True
    
    print('측정 시작')
    GPIO.output(14, True)
    shared_data.data_store['Sample_pump'] = True
    
    GPIO.output(15, True)
    shared_data.data_store['Solution1_pump'] = True
    
    GPIO.output(25, True)
    shared_data.data_store['Solution2_pump'] = True
    
    time.sleep(60)

    GPIO.output(18, False)  # 탄소 펌프 종료
    shared_data.data_store['Carbon_pump'] = False
    print('탄소 펌프 완료')

    time.sleep(240)

    GPIO.output(27, False)  # 유입 펌프 종료
    shared_data.data_store['Input_pump'] = False
    print('유입 펌프 완료')

    time.sleep(600)

    GPIO.output(14, False)
    shared_data.data_store['Sample_pump'] = False
    
    GPIO.output(15, False)
    shared_data.data_store['Solution1_pump'] = False
    
    GPIO.output(25, False)
    shared_data.data_store['Solution2_pump'] = False
    print('측정 펌프 완료')

    # 남은 시간 동안 대기
    time.sleep(1200)

    # 공기 펌프 시작
    print('공기 펌프 시작')
    GPIO.output(23, True)  # Air_pump 시작
    shared_data.data_store['Air_pump'] = True
    
    previous_data = aeration_adjust.get_previous_cycle_data()
    aeration_time = aeration_adjust.adjust_aeration_time(previous_data)  # 폭기 시간을 조절하여 가져옴

    time.sleep(aeration_time)  # 조정된 폭기 시간 동안 대기

    GPIO.output(23, False)  # Air_pump 종료
    shared_data.data_store['Air_pump'] = False
    print('공기 펌프 완료')
    
    GPIO.output(24, False)  # Mixer 시작
    shared_data.data_store['Mixer'] = False
    print('교반 완료')
    
    rest_time = 900 - aeration_time
    print(f"Remaining time before aeration: {rest_time} seconds")
    time.sleep(rest_time)

    print("300초 동안 센서 데이터 수집 시작")
    # 300초간 데이터 수집 후 평균 계산
    avg_clearlight, avg_red, avg_green, avg_blue = sensor_logging.collect_and_average_sensor_data(300)
    
    print(f"평균 센서 데이터: Clearlight={avg_clearlight}, Red={avg_red}, Green={avg_green}, Blue={avg_blue}")
    
    # 첫 번째 사이클 여부에 따른 폭기 시간 기록
    sensor_logging.log_sensor_data_to_csv(avg_clearlight, avg_red, avg_green, avg_blue, concentration=None, aeration_time=aeration_time)

    GPIO.output(22, True)  # 방류 펌프 시작
    shared_data.data_store['Output_pump'] = True
    print('방류 펌프 시작')

    time.sleep(290)

    GPIO.output(22, False)  # 방류 펌프 종료
    shared_data.data_store['Output_pump'] = False
    print('방류 완료')
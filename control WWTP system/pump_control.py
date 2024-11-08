import RPi.GPIO as GPIO
import shared_data
import time
import os

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

# 펌프 상태를 지속적으로 업데이트
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

    time.sleep(1200)

    # 공기 펌프 시작
    print('공기 펌프 시작')
    GPIO.output(23, True)  # Air_pump 시작
    shared_data.data_store['Air_pump'] = True

    time.sleep(900)  # 조정된 폭기 시간 동안 대기

    GPIO.output(23, False)  # Air_pump 종료
    shared_data.data_store['Air_pump'] = False
    print('공기 펌프 완료')
    
    GPIO.output(24, False)  # Mixer 시작
    shared_data.data_store['Mixer'] = False
    print('교반 완료')

    time.sleep(300)

    GPIO.output(22, True)  # 방류 펌프 시작
    shared_data.data_store['Output_pump'] = True
    print('방류 펌프 시작')

    time.sleep(290)

    GPIO.output(22, False)  # 방류 펌프 종료
    shared_data.data_store['Output_pump'] = False
    print('방류 완료')
import csv
import os
import shared_data
import time

# 센서 평균값, 농도, 폭기 시간 파일을 저장할 경로 설정
average_data_file = "/home/pi/wwtp/sensor_concentration/average_sensor_data.csv"

def log_sensor_data_to_csv(avg_clearlight, avg_red, avg_green, avg_blue, concentration=None, aeration_time=None):
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(average_data_file), exist_ok=True)
    
    # CSV 파일이 존재하지 않으면 새로 생성
    file_exists = os.path.isfile(average_data_file)
    
    try:
        with open(average_data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # 파일이 새로 생성된 경우 헤더 추가
            if not file_exists:
                writer.writerow(["Timestamp", "Average_Clearlight", "Average_Red", "Average_Green", "Average_Blue", "Concentration", "Aeration_Time"])
            
            # main.py에서 생성된 타임스탬프 사용
            timestamp = shared_data.timestamp  # 공유된 타임스탬프 사용
            
            # 기록할 값 확인
            print(f"Logging to CSV: {timestamp}, {avg_clearlight}, {avg_red}, {avg_green}, {avg_blue}, {concentration}, {aeration_time}")

            # CSV에 데이터 추가 (concentration과 aeration_time을 0으로 처리)
            writer.writerow([
                timestamp, 
                avg_clearlight, 
                avg_red, 
                avg_green, 
                avg_blue, 
                concentration,
                aeration_time if aeration_time is not None else float(0.0)    # 공란 대신 0 처리
            ])
        
        print(f"Data logged to {average_data_file}")
    except Exception as e:
        print(f"Error logging sensor data: {e}")


def collect_and_average_sensor_data(duration):
    clearlight_vals = []
    red_vals = []
    green_vals = []
    blue_vals = []

    start_time = time.time()
    while time.time() - start_time < duration:
        # 센서 데이터를 shared_data에서 읽어옴
        clearlight_vals.append(shared_data.data_store['Clearlight'] or 0)
        red_vals.append(shared_data.data_store['Red'] or 0)
        green_vals.append(shared_data.data_store['Green'] or 0)
        blue_vals.append(shared_data.data_store['Blue'] or 0)

        time.sleep(1)  # 1초마다 데이터 수집

    avg_clearlight = sum(clearlight_vals) / len(clearlight_vals) if clearlight_vals else 0
    avg_red = sum(red_vals) / len(red_vals) if red_vals else 0
    avg_green = sum(green_vals) / len(green_vals) if green_vals else 0
    avg_blue = sum(blue_vals) / len(blue_vals) if blue_vals else 0

    return avg_clearlight, avg_red, avg_green, avg_blue

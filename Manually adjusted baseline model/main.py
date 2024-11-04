import threading
import time
import os
import RPi.GPIO as GPIO
import shared_data
import pump_control  # import pump_control  # ㅎ프 제어 모듈
import ammonia_detect  # 암모니아 데이터 수집
import app  # Flask 웹 서버 (농도 입력받는 기능)
import webbrowser  # 웹 브라우저 자동 실행을 위한 모듈
import csv

# 실시간 센서 및 팔프 상태 데이터를 저장할 CSV 파일 설정
csv_folder = "/home/pi/wwtp/data"
csv_file = None
csv_writer = None
csv_lock = threading.Lock()

os.environ['DISPLAY'] = ':0'

# 전역 변수로 파일 이름 공용
shared_data.csv_file_path = None
shared_data.first_cycle = True  # 첫 번째 사이클 여부를 관리하는 플래그

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
    

def create_new_csv_file():
    global csv_file, csv_writer
    # 파일이 이미 열려 있는 경우 새로운 파일을 생성하지 않음
    if csv_file is not None:
        return
    timestamp = shared_data.timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_path = os.path.join(csv_folder, f"{timestamp}.csv")
    shared_data.timestamp = timestamp

    try:
        with csv_lock:
            print(f"Creating new CSV file: {csv_file_path}")
            csv_file = open(csv_file_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            # CSV 파일 헤더 작성
            csv_writer.writerow(["Timestamp", "Mixer", "Input_pump", "Carbon_pump", "Air_pump", "Output_pump",
                                 "Sample_pump", "Solution1_pump", "Solution2_pump", "Clearlight", "Red", "Green", "Blue"])
            print(f"Created new CSV file: {csv_file_path}")
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        csv_file = None
        csv_writer = None

# 실시간 데이터 기록
def log_realtime_data():
    global csv_writer, csv_file

    # CSV 파일이 없는 경우 생성
    if csv_writer is None:
        create_new_csv_file()

    while not shared_data.stop_thread:
        try:
            # CSV 작성 시 데이터 행 구성
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            data_row = [
                timestamp,
                shared_data.data_store['Mixer'],
                shared_data.data_store['Input_pump'],
                shared_data.data_store['Carbon_pump'],
                shared_data.data_store['Air_pump'],
                shared_data.data_store['Output_pump'],
                shared_data.data_store['Sample_pump'],
                shared_data.data_store['Solution1_pump'],
                shared_data.data_store['Solution2_pump'],
                shared_data.data_store['Clearlight'] if shared_data.data_store['Clearlight'] is not None else "",
                shared_data.data_store['Red'] if shared_data.data_store['Red'] is not None else "",
                shared_data.data_store['Green'] if shared_data.data_store['Green'] is not None else "",
                shared_data.data_store['Blue'] if shared_data.data_store['Blue'] is not None else ""
            ]

            # CSV 파일에 데이터 기록
            with csv_lock:
                if csv_writer is not None:
                    csv_writer.writerow(data_row)
                    csv_file.flush()  # 버퍼 비우기

            print(f"[INFO] Logging real-time data: {data_row}")
            time.sleep(1)  # 1초마다 데이터 기록

        except Exception as e:
            print(f"[ERROR] Error logging real-time data: {e}")
            break  # 예외 발생 시 루프 종료

# 실시간 CSV 파일 종료
def close_csv_file():
    global csv_file, csv_writer
    try:
        with csv_lock:
            if csv_file:
                csv_file.flush()  # 파일을 강제로 디스크에 기록
                csv_file.close()
            csv_file = None
            csv_writer = None
            print("Closed CSV file")
    except Exception as e:
        print(f"Error closing CSV file: {e}")

# 팔프 실행 함수
def run_pump():
    pump_control.setup_pins()  # GPIO 핀 설정
    try:
        print("Starting pump control process")
        pump_control.run_pump_sequence()  # pump_control.py에 정의된 팔프 제어 함수 실행
        # 프로그램 종료 시 파일을 반드시 닫기
        close_csv_file()
        print("Rebooting system...")
        time.sleep(1)
        os.system('sudo reboot')  # 시스템 재부팅 명령
    except Exception as e:
        print(f"Error during pump operation: {e}")
    finally:
        # 프로그램 종료 시 파일을 반드시 닫기
        close_csv_file()

# Flask 서버를 300초 후에 실행하는 함수
def delayed_flask_start():
    print("5분 후 Flask 서버를 실행합니다...")
    time.sleep(300)  # 300초 대기
    flask_thread = threading.Thread(target=app.start_flask)
    flask_thread.daemon = True  # Flask 서버가 백그라운드에서 실행되도록 설정
    flask_thread.start()

    # Flask 서버 시작 후 웹 브라우저 열기
    webbrowser.open("http://localhost:5000")

def run_main():
    
    # GPIO 핀 설정
    pump_control.setup_pins()

    # 첫 번째 루프인지 확인 (CSV 파일 기반)
    if pump_control.is_first_cycle():
        print("첫 번째 루프입니다. Flask 서버를 생략합니다.")
    else:
        # 첫 번째 루프가 아닌 경우 Flask 서버 실행
        flask_delay_thread = threading.Thread(target=delayed_flask_start)
        flask_delay_thread.start()
        print("Flask 서버 실행 중... 농도를 입력할 수 있습니다.")

    # 암모니아 데이터 수집 스레드 실행
    ammonia_thread = threading.Thread(target=ammonia_detect.main_loop)
    ammonia_thread.start()

    # 팔프 실행 스레드 시작
    pump_thread = threading.Thread(target=run_pump)
    pump_thread.start()

    # 실시간 데이터 기록 스레드 시작
    log_thread = threading.Thread(target=log_realtime_data)
    log_thread.start()

    # 모든 스레드 종료 대기
    pump_thread.join()
    log_thread.join()

    # 프로그램 종료 시 CSV 파일 닫기
    close_csv_file()

if __name__ == "__main__":
    try:
        run_main()
    except KeyboardInterrupt:
        print("프로그램 중단")
    finally:
        shared_data.stop_thread = True
        close_csv_file()  # 프로그램 종료 시 CSV 파일 닫기
        GPIO.cleanup()
        print("GPIO 설정 초기화 완료")
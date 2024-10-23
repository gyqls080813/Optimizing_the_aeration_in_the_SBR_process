import datetime
import time
import serial
from serial.tools import list_ports
import threading
import shared_data

ser = None
PORT = None
start_time = datetime.datetime.now()

def auto_select_arduino_port():
    ports = list(list_ports.comports())
    print("Available ports:", ports)  # 디버깅을 위해 사용 가능한 포트 출력
    for port in ports:
        if 'ttyACM0' in port.description:
            return port.device
    return None

def setup_and_start():
    global PORT, ser

    PORT = auto_select_arduino_port()

    if PORT is None:
        print("[Error] No Arduino port found. Please check the connection.")
        return False

    try:
        ser = serial.Serial(PORT, 9600, timeout=1)
        print("[Success] " + PORT + " is selected as the COM Port.")
        return True
    except serial.SerialException as e:
        print(f"[Error] Could not open serial port {PORT}: {e}")
        return False

def get_data():
    global ser

    ser.write('1'.encode('utf-8'))

    if ser.readable():
        try:
            data_str = ser.readline().decode().strip()

            # Process the data if not empty
            if data_str:
                tmp_list = data_str.split(",")
                if len(tmp_list) >= 4:  # Ensures there are enough data points
                    shared_data.data_store['Clearlight'], shared_data.data_store['Red'], shared_data.data_store['Green'], shared_data.data_store['Blue'] = map(float, tmp_list[:4])
                    return shared_data.data_store['Clearlight'], shared_data.data_store['Red'], shared_data.data_store['Green'], shared_data.data_store['Blue']
                else:
                    print("Incomplete data received.")
            else:
                print("No data received.")
        except ValueError as e:
            print(f"Error processing data: {e}")

    ser.write('0'.encode('utf-8'))
    return None

def main_loop():
    global start_time
    if setup_and_start():
        start_time = datetime.datetime.now()  # 데이터 수집 시작 시간 설정
        while True:
            get_data()
            time.sleep(1)
    else:
        print("[Error] Failed to set up and start.")
        return None

if __name__ == "__main__":
    main_loop()

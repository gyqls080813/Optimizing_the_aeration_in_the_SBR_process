import os
import csv

# CSV 파일에서 이전 사이클 데이터를 가져오는 함수
def get_previous_cycle_data(csv_file="/home/pi/wwtp/sensor_concentration/average_sensor_data.csv"):
    # CSV 파일이 존재하는지 확인
    if not os.path.exists(csv_file):
        print(f"[ERROR] No data available. The file {csv_file} does not exist.")
        return {'c': 0, 'r': 0, 'g': 0, 'b': 0, 'aeration_time': 900}  # 기본값 반환

    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

            # 데이터가 충분하지 않은 경우 처리
            if len(rows) < 2:
                print("[WARNING] Not enough data: less than 2 cycles available.")
                return {'c': 0, 'r': 0, 'g': 0, 'b': 0, 'aeration_time': 900}  # 기본값 반환

            # 마지막 행의 데이터를 가져오기
            previous_row = rows[-1]  # 마지막 사이클

            # 이전 사이클의 센서 데이터 가져오기
            avg_clearlight = float(previous_row.get('Average_Clearlight', 0))
            avg_red = float(previous_row.get('Average_Red', 0))
            avg_green = float(previous_row.get('Average_Green', 0))
            avg_blue = float(previous_row.get('Average_Blue', 0))
            aeration_time = float(previous_row.get('Aeration_Time',900))

            # 가져온 데이터 출력
            print(f"[INFO] Previous cycle data: Clearlight={avg_clearlight}, Red={avg_red}, Green={avg_green}, Blue={avg_blue}, Aeration Time={aeration_time}")
            return {'c': avg_clearlight, 'r': avg_red, 'g': avg_green, 'b': avg_blue, 'aeration_time': aeration_time}

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
    except ValueError as e:
        print(f"[ERROR] Value error while parsing previous cycle data: {e}")
    except KeyError as e:
        print(f"[ERROR] Missing key in previous cycle data: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

    # 예외 발생 시 기본값 반환
    return {'c': 0, 'r': 0, 'g': 0, 'b': 0, 'aeration_time': 900}

def adjust_aeration_time(previous_data):
    
    try:
        base_time = float(previous_data.get('aeration_time', 0) or 0)  # 공란이나 None일 때 0으로 처리
    except ValueError:
        base_time = 0  # 변환 실패 시 기본값 0
    
    try:
        c = previous_data['c']
        if c > 660:
            base_time = max(0, base_time - 10)  # 1초 줄이되, 최소 0초 유지
        elif c <= 660:
            base_time = min(900, base_time + 10)  # 1초 늘리되, 최대 10초 유지

        # base_time이 0보다 작으면 0으로 저장하도록 예외 처리
        if base_time <= 0:
            base_time = 0  # 공란 대신 0으로 저장

        print(f"[INFO] Adjusted aeration time: {base_time} seconds")

    except (ValueError, KeyError) as e:
        print(f"[ERROR] Error adjusting aeration time: {e}")

    return base_time


if __name__ == "__main__":
    previous_data = get_previous_cycle_data()
    aeration_time = adjust_aeration_time(previous_data)
    print(f"Final aeration time for the next cycle: {aeration_time} seconds")

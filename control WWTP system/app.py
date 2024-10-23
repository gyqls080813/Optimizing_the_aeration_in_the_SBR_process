from flask import Flask, request, render_template
import csv
import os

app = Flask(__name__)

# CSV 파일 경로 설정
average_data_file = "/home/pi/wwtp/sensor_concentration/average_sensor_data.csv"

# 농도를 입력받아 이전 사이클에 기록
def update_previous_cycle_with_concentration(concentration):
    rows = []
    
    # CSV 파일이 존재하는지 확인
    if not os.path.exists(average_data_file):
        print(f"Error: {average_data_file} 파일이 존재하지 않습니다.")
        return
    
    try:
        # 파일 읽기
        with open(average_data_file, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if len(rows) > 1:  # 데이터가 있어야 함 (헤더 제외)
            print(f"Updating concentration for last cycle with value: {concentration}")
            rows[-1][-2] = concentration  # 마지막 행의 농도 값 업데이트 (Concentration 열)
        
        # 파일 쓰기 (수정된 데이터를 다시 저장)
        with open(average_data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    except Exception as e:
        print(f"Error updating concentration: {e}")

# Flask 서버 (농도 입력 받기)
@app.route('/', methods=['GET', 'POST'])
def concentration_input():
    if request.method == 'POST':
        concentration = request.form['concentration']
        update_previous_cycle_with_concentration(concentration)
        return f"Concentration {concentration} saved for the previous cycle!"
    return render_template('index.html')

# Flask 서버를 시작하는 함수
def start_flask():
    app.run(host='0.0.0.0', port=5000)

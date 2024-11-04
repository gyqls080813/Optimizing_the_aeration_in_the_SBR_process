# 공유 데이터 초기화
data_store = {
    'Clearlight': None,
    'Red': None,
    'Green': None,
    'Blue': None,
    'Sample_pump': False,
    'Solution1_pump': False,
    'Solution2_pump': False,
    'Mixer': False,
    'Input_pump': False,
    'Carbon_pump': False,
    'Air_pump': False,
    'Output_pump': False
}

# 스레드 실행 제어용 변수
stop_thread = False
timestamp = None  # main.py에서 생성된 값을 공유
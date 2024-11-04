# Advanced wastewater treatment

<img src="./Image/SBR process.png" align="right" width="40%"/>
<br><br><br><br><br><br><br>
Advanced wastewater treatment is a highly modernized process designed to efficiently remove nitrogen and phosphorus from wastewater. Among these methods, the SBR (Sequence Batch Reactor) process allows for sequential water treatment in batch reactions, enabling precise time-based control. <br><br>
In this study, we optimized the aeration time in the SBR wastewater treatment process. To achieve this, we adjusted the aeration period to reach a specific level of nitrification before stopping aeration. The extent of nitrification was monitored by measuring the ammonia concentration in the effluent using an RGB sensor.
<br><br><br><br><br><br><br>

# Code Guide
The 'Optimizing the Aeration in the SBR Process' repository contains code aimed at achieving this goal. Each code module is designed to operate using the Raspberry Pi 2. The code is structured as follows:
<br>
1. control WWTP system
2. Manually adjusted baseline model
3. Controlled Reinforcement Learning(DQN) model
<br>
Each code module serves the following functions:<br><br><br>
1. control WWTP system<br><br>
<img src="./Image/sbr automatic system.png" align="right" width="50%"/>
First, the control WWTP system code is used to control the SBR process with a Raspberry Pi 2. The pump and stirring mechanisms are controlled via GPIO connections. The details for each part of the code are as follows:<br><br><br>
1.1 main.py<br><br>
모든 코드를 종합하여 실행하는 파이썬 파일입니다. shared_data.py에 종합적인 데이터가 입력되면 해당 데이터를 기반으로 RGB sensor 데이터를 csv파일에 기록 및 저장합니다. 또한 pump_control.py에 작성된 것과 같이 SBR system을 작동시킵니다. 또한 1cycle이 진행되면 Raspberry Pi를 재부팅시켜 오류를 방지합니다.<br><br><br>
1.2 ammonia_detect.py<br><br>
아두이노 포트를 관리하여 정상적으로 연결하도록 기능한다. 또한 RGB 센서와 연결되어 들어오는 RGB 데이터를 받아오는 역할을 한다. 이렇게 들어온 데이터는 shared_data.py에 전송되어 csv 파일에 작성되도록 한다.<br><br><br>
1.3 pump_control.py<br><br>
GPIO와 연결된 기기들을 작동시키는 역할을 한다. 각 GPIO는 Raspberry Pi2에 알맞는 포트를 참고해서 연결시켰다. 또한 shared_data.py에 실시간 펌프 상태를 전송시키는 역할을 한다. 해당 파일에서는 1hour로 최적화된 SBR 공정에 맞게 작성되었다. 따라서 SBR process를 자동화 시키기를 원한다면, 시간과 GPIO를 고려하여 수정 후 사용하면 된다.<br><br><br>
1.4 shared_data.py<br><br>
데이터를 종합하는 파이썬 파일이다. ammonia_detect.py와 pump_control.py에서 받아온 데이터를 종합하고, main.py에 보내 csv파일에 작성하도록 한다. <br><br><br>
2. Manually adjusted baseline model<br><br>
<img src="./Image/Manually adjusted baseline model.png" align="right" width="50%"/>
1.1 main.py<br><br>
control WWTP system의 main.py와 동일한 역할을 한다. 이때 flask를 열어 해당 cycle의 ammonia concentration을 입력받는 코드가 포함된다.<br><br><br>
1.2 ammonia_detect.py<br><br>
control WWTP system의 ammonia_detect.py와 동일한 역할을 한다.<br><br><br>
1.2 sensor_logging.py<br><br>
control WWTP system의 ammonia_detect.py와 동일한 역할을 한다.<br><br><br>
1.3 pump_control.py<br><br>
control WWTP system의 pump_control.py와 동일한 역할을 한다. 추가적인 기능은 aeration time 결정이다. 이는 aeration_adjust.py에 의해 결정된다. 또한 csv 파일을 작성하는 코드도 포함되어 있다. <br><br>
1.4 shared_data.py<br><br>
control WWTP system의 shared_data.py와 동일한 역할을 한다. timestamp = None를 이용하여 농도에 따른 센서값을 추가하도록 변수를 추가했다. <br><br>
1.5 sensor_logging.py<br><br>
펌프가 모두 작동된 후 침전 시간 중 5분동안 센서 데이터를 저장한 뒤 이를 평균내여 average_sensor_data.csv에 저장한다. 해당 함수는 pump_control.py에서 사용되며 측정 시간을 입력하여 더 많은 데이터 <br><br>
1.6 aeration_adjust.py<br><br>
control WWTP system의 pump_control.py와 동일한 역할을 한다. 추가적인 기능은 aeration time 결정이다. 이는 aeration_adjust.py에 의해 결정된다. 이때 sensor data의 기준은 매번 1mg/L를 측정하여 선택했다. 해당 연구의 수동 조절이라는 의미는 해당 기준을 직접 측정하여 변화시킨 것에 있다. <br><br><br>
3. Controlled Reinforcement Learning(DQN) model<br><br>
<img src="./Image/DQN simulation.png" align="right" width="50%"/>
1.1 main.py<br><br>
control WWTP system의 main.py와 동일한 역할을 한다. 이때 flask를 열어 해당 cycle의 ammonia concentration을 입력받는 코드가 포함된다.<br><br><br>
1.2 ammonia_detect.py<br><br>
control WWTP system의 ammonia_detect.py와 동일한 역할을 한다.<br><br><br>
1.3 pump_control.py<br><br>
control WWTP system의 pump_control.py와 동일한 역할을 한다. 단, aeration time은 aeration_adjust.py에 의해 결정된다. <br><br><br>
1.4 shared_data.py<br><br>
데이터를 종합하는 파이썬 파일이다. ammonia_detect.py와 pump_control.py에서 받아온 데이터를 종합하고, main.py에 보내 csv파일에 작성하도록 한다. <br><br><br>
<img src="./Image/log.svg" align="left" width="60%"/><br><br><br>
Minyeop Lee<br><br>
Affiliation.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Eco Friendly Bio Lab<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Division of Environmental Energy Engineering<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Yonsei University<br><br>
Location.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Baekwoon Hall, Room 123<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1, Yonseidae-gil<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Wonju, Gangwon Special Self-Governing Province<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; South Korea<br><br>
Contact.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; E-mail	: gyqls0808133@gmail.com<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub	: github.com/gyqls080813<br>



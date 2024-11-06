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
This is the main Python file that integrates all the code. When comprehensive data is input into shared_data.py, it records and saves RGB sensor data into a CSV file based on that data. Additionally, it operates the SBR system as described in pump_control.py. After one cycle is completed, it restarts the Raspberry Pi to prevent errors.<br><br><br>
1.2 ammonia_detect.py<br><br>
This file manages the Arduino port to ensure proper connection. It also connects to the RGB sensor to receive incoming RGB data. This data is sent to shared_data.py to be logged into the CSV file.<br><br><br>
1.3 pump_control.py<br><br>
This file operates devices connected to GPIO. Each GPIO is connected according to the appropriate ports for Raspberry Pi 2. It also sends real-time pump status to shared_data.py. This file is optimized for the SBR process set to one hour. Therefore, if you want to automate the SBR process, you can modify it considering time and GPIO.<br><br><br>
1.4 shared_data.py<br><br>
This is the Python file that aggregates data. It consolidates data received from ammonia_detect.py and pump_control.py and sends it to main.py for logging into the CSV file.<br><br><br>
2. Manually adjusted baseline model<br><br>
<img src="./Image/Manually adjusted baseline model.png" align="right" width="50%"/>
2.1 main.py<br><br>
This file serves the same function as the main.py of the control WWTP system. It includes code to open Flask and receive input for ammonia concentration for the cycle.<br><br><br>
2.2 ammonia_detect.py<br><br>
This file performs the same role as ammonia_detect.py in the control WWTP system.<br><br><br>
2.3 sensor_logging.py<br><br>
This file serves the same function as ammonia_detect.py in the control WWTP system.<br><br><br>
2.4 pump_control.py<br><br>
This file functions the same as pump_control.py in the control WWTP system. An additional feature is determining aeration time, which is established by aeration_adjust.py. It also includes code to write to the CSV file.<br><br><br>
2.5 shared_data.py<br><br>
This file serves the same role as shared_data.py in the control WWTP system. It has been modified to add a variable to include sensor values based on concentration using timestamp = None.<br><br><br>
2.6 sensor_logging.py<br><br>
After all pumps have been activated, this file saves sensor data for five minutes during the settling time and averages it before saving it to average_sensor_data.csv. This function is utilized in pump_control.py to enter measurement times for collecting more data.<br><br><br>
2.7 aeration_adjust.py<br><br>
This file serves the same role as pump_control.py in the control WWTP system. An additional feature is determining aeration time, which is established by aeration_adjust.py. For this, the standard for sensor data is measured at 1 mg/L each time. The term "manual adjustment" in this study refers to the practice of measuring and altering this standard directly.<br><br><br>
4. Controlled Reinforcement Learning(DQN) model<br>
<img src="./Image/DQN simulation.png" align="mid" width="100%"/>
The section on Controlled Reinforcement Learning (DQN) model currently only has simulation code implemented. It was created to see how well the DQN model copes with an environment similar to the actual one.<br><br><br>
3.1 SBR_Environment.py<br><br>
This file simulates the actual SBR environment we use. The incoming concentration is randomly specified every 50 times. In the case of a stabilized cycle of 1L, the inflow and existing volume are diluted in a 1:23 ratio. Thus, the inflow is calculated as 1, and the existing amount as 23 to determine the starting ammonia concentration. An equation for the effluent ammonia based on aeration time and incoming ammonia concentration has also been created.<br><br><br>
3.2 Timegan.py<br><br>
This file has been trained using data from the actual SBR reactor operation. The training was conducted for 10,000 epochs. The data located in ./data is an example of the trained data. Only data from 3300 to 3600 were used for the training.<br><br><br>
3.3 generation_timedata.py<br><br>
This section loads the trained Timegan to generate time-series data for desired concentrations. The trained model is located in ./model/timegan.<br><br><br>
3.4 main_model.py<br><br>
This is the model we used. This code includes both the reinforcement learning model (DQN) and the concentration prediction model (MLP).<br><br><br>
3.5 mlp_regression_pretrain.py<br><br>
This is the code that trained the MLP model. The training data is the same as that from Timegan. The number of epochs was set to 10,000.<br><br><br>
3.6 main.py<br><br>
This is the file that operates all the code. It loads the models and determines the reward structure. In this case, for the first and second cycles, the aeration time is specified before operation. This is because, in the actual environment, there is no previous data for these two cycles, so a fixed value of 900 is applied.<br><br><br><br><br><br><br>
<img src="./Image/log.png" align="mid" width="100%"/>



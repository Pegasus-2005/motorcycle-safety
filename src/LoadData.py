import pandas as pd
import numpy as np
import os

"""
SENSOR SIGNALS 

+───────────────────────────+───────────────────────────────────────────────+───────────────────+─────+
| signal                    | description                                   | unit              |comp.|
+───────────────────────────+───────────────────────────────────────────────+───────────────────+─────+
| Time                      | time                                          | s                 |     |

                                                MOTORCYCLE BODY 
| MotoBody_angaccRes        | motorcycle body angular acceleration          | radian per s^2    |     |
| MotoBody_angaccX          | motorcycle body angular acceleration          | radian per s^2    |  x  |
| MotoBody_angaccY          | motorcycle body angular acceleration          | radian per s^2    |  y  |
| MotoBody_angaccZ          | motorcycle body angular acceleration          | radian per s^2    |  z  |
| MotoBody_angvelRes        | motorcycle body linear velocity               | radian per s      |     |
| MotoBody_angvelX          | motorcycle body linear velocity               | radian per s      |  x  |
| MotoBody_angvelY          | motorcycle body linear velocity               | radian per s      |  y  |
| MotoBody_angvelZ          | motorcycle body linear velocity               | radian per s      |  z  |
| MotoBody_linaccRes        | motorcycle body linear acceleration           | m per s^2         |     |
| MotoBody_linaccX          | motorcycle body linear acceleration           | m per s^2         |  x  |
| MotoBody_linaccY          | motorcycle body linear acceleration           | m per s^2         |  y  |
| MotoBody_linaccZ          | motorcycle body linear acceleration           | m per s^2         |  z  |
| MotoBody_linvelRes        | motorcycle body linear velocity               | m per s           |     |
| MotoBody_linvelX          | motorcycle body linear velocity               | m per s           |  x  |
| MotoBody_linvelY          | motorcycle body linear velocity               | m per s           |  y  |
| MotoBody_linvelZ          | motorcycle body linear velocity               | m per s           |  z  |

                                                MOTORCYCLE REAR WHEEL 
| MotoRW_linaccRes          | motorcycle rear wheel linear acceleration     | m per s^2         |     |
| MotoRW_linaccX            | motorcycle rear wheel linear acceleration     | m per s^2         |  x  |
| MotoRW_linaccY            | motorcycle rear wheel linear acceleration     | m per s^2         |  y  |
| MotoRW_linaccZ            | motorcycle rear wheel linear acceleration     | m per s^2         |  z  |
| MotoRW_linvelRes          | motorcycle rear wheel linear velocity         | m per s           |     |
| MotoRW_linvelX            | motorcycle rear wheel linear velocity         | m per s           |  x  |
| MotoRW_linvelY            | motorcycle rear wheel linear velocity         | m per s           |  y  |
| MotoRW_linvelZ            | motorcycle rear wheel linear velocity         | m per s           |  z  |
| RW_angacc_Y               | rear wheel angular acceleration               | radian per s^2    |  y  |
| RW_angvel_Y               | rear wheel angular velocity                   | radian per s      |  y  |
| RW_Road_cnt_force         |  rear wheel road contact force                | N                 |     |
| RW_Car_cnt_force          |  rear wheel car contact force                 | N                 |     |

                                                MOTORCYCLE FRONT WHEEL 
| MotoFW_linaccRes          | motorcycle front wheel linear acceleration    | m per s^2         |     |
| MotoFW_linaccX            | motorcycle front wheel linear acceleration    | m per s^2         |  x  |
| MotoFW_linaccY            | motorcycle front wheel linear acceleration    | m per s^2         |  y  |
| MotoFW_linaccZ            | motorcycle front wheel linear acceleration    | m per s^2         |  z  |
| MotoFW_linvelRes          | motorcycle front wheel linear velocity        | m per s           |     |
| MotoFW_linvelX            | motorcycle front wheel linear velocity        | m per s           |  x  |
| MotoFW_linvelY            | motorcycle front wheel linear velocity        | m per s           |  y  |
| MotoFW_linvelZ            | motorcycle front wheel linear velocity        | m per s           |  z  |
| FW_angacc_Y               | front wheel angular acceleration              | radian per s^2    |  y  |
| FW_angvel_Y               | front wheel angular velocity                  | radian per s      |  y  |
| FW_Road_cnt_force         |  front wheel road contact force               | N                 |     |
| FW_Car_cnt_force          |  front wheel car contact force                | N                 |     |

                                                FRONT DEFLECTION 
| FrontDef_angdispX         | front deflection angular position             | radian            |  x  |
| FrontDef_angdispY         | front deflection angular position             | radian            |  y  |
| FrontDef_angdispZ         | front deflection angular position             | radian            |  z  |
| Frontdef_angacc           | front deflection angular acceleration         | radian per s^2    |     |
| Frontdef_angvel           | front deflection angular velocity             | radian per s      |     |
| Frontdef_angpos           | front deflection angular position             | radian            |     |

                                                FRONT SUSPENSION 
| FrontSuspension_linacc    | front suspension linear acceleration          | m per s^2         |     |
| FrontSuspension_linvel    | front suspension linear velocity              | m per s           |     |
| FrontSuspension_linpos    | front suspension linear position              | m                 |     |

                                                REAR SUSPENSION 
| RearSuspension_angacc     | rear suspension angular acceleration          | radian per s^2    |     |
| RearSuspension_angvel     | rear suspension angular velocity              | radian per s      |     |
| RearSuspension_angpos     | rear suspension angular position              | radian            |     |

                                                ROATIONAL JOINTS 
| JointFW_angacc            |  angular acceleration of front wheel joint    | radian per s^2    |     |
| JointFW_angvel            |  angular velocity of front wheel joint        | radian per s      |     |
| JointRW_angacc            |  angular acceleration of rear wheel joint     | radian per s^2    |     |
| JointRW_angvel            |  angular velocity of rear wheel joint         | radian per s      |     |

                                                CRASH LABEL 
| CrashLabelML              |  crash label                                  | 0 or 1            |     |
+──────────────────────────────────+────────────────────────────────────────+───────────────────+─────+
"""

# list of all available features, comment out the ones you don't want to use
features = [
    'Time',
    'MotoBody_angaccRes',
    'MotoBody_angaccX',
    'MotoBody_angaccY',
    'MotoBody_angaccZ',
    'FrontDef_angdispX',
    'FrontDef_angdispY',
    'FrontDef_angdispZ',
    'MotoBody_angvelRes',
    'MotoBody_angvelX',
    'MotoBody_angvelY',
    'MotoBody_angvelZ',
    'MotoRW_linaccRes',
    'MotoRW_linaccX',
    'MotoRW_linaccY',
    'MotoRW_linaccZ',
    'MotoFW_linaccRes',
    'MotoFW_linaccX',
    'MotoFW_linaccY',
    'MotoFW_linaccZ',
    'MotoBody_linaccRes',
    'MotoBody_linaccX',
    'MotoBody_linaccY',
    'MotoBody_linaccZ',
    'MotoFW_linvelRes',
    'MotoFW_linvelX',
    'MotoFW_linvelY',
    'MotoFW_linvelZ',
    'MotoRW_linvelRes',
    'MotoRW_linvelX',
    'MotoRW_linvelY',
    'MotoRW_linvelZ',
    'MotoBody_linvelRes',
    'MotoBody_linvelX',
    'MotoBody_linvelY',
    'MotoBody_linvelZ',
    'Frontdef_angacc',
    'Frontdef_angpos',
    'JointRW_angvel',
    'Frontdef_angvel',
    'JointFW_angvel',
    'SwitchSidesensor_left_car',
    'SwitchSidesensor_right_road',
    'SwitchSidesensor_right_car',
    'Sensor_left_road',
    'Sensor_left_car',
    'Sensor_right_road',
    'Sensor_right_car',
    'CrashLabelML',
    'SwitchSidesensor_left_road'
]


# %% load raw data
datapath = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# train data
data_train = pd.read_csv(os.path.join(datapath, 'trainingData.csv'))
data_train['angveldiff'] = data_train.FW_angvel_Y - data_train.RW_angvel_Y
data_train['angaccdiff'] = data_train.FW_angacc_Y - data_train.RW_angacc_Y
data_train['sensorLeft'] = np.logical_or(data_train.SwitchSidesensor_left_road, data_train.SwitchSidesensor_left_car)
data_train['sensorRight'] = np.logical_or(data_train.SwitchSidesensor_right_road, data_train.SwitchSidesensor_right_car)
data_train['FW_cnt_Force'] = data_train.FW_Car_cnt_force + data_train.FW_Road_cnt_force
data_train['RW_cnt_Force'] = data_train.RW_Car_cnt_force + data_train.RW_Road_cnt_force

# test data
data_test = pd.read_csv(os.path.join(datapath, 'testData.csv'))
data_test['angveldiff'] = data_test.FW_angvel_Y - data_test.RW_angvel_Y
data_test['angaccdiff'] = data_test.FW_angacc_Y - data_test.RW_angacc_Y
data_test['sensorLeft'] = np.logical_or(data_test.SwitchSidesensor_left_road, data_test.SwitchSidesensor_left_car)
data_test['sensorRight'] = np.logical_or(data_test.SwitchSidesensor_right_road, data_test.SwitchSidesensor_right_car)
data_test['FW_cnt_Force'] = data_test.FW_Car_cnt_force + data_test.FW_Road_cnt_force
data_test['RW_cnt_Force'] = data_test.RW_Car_cnt_force + data_test.RW_Road_cnt_force


# %% select features and split into train and test data

# input data consists of selected features
X_train = data_train[features].to_numpy()
# output data consists of crash label
y_train = data_train[['CrashLabelML']].to_numpy()

# input data consists of selected features
X_train = data_test[features].to_numpy()
# output data consists of crash label
y_train = data_test[['CrashLabelML']].to_numpy()


# %% special scenarios including iso scenarios
test_paths = [
    # NON CRASH
    'Curbstone_5_Out.csv',
    'Pothole_5_Out.csv',
    'Randomstate_100_Out.csv',
    'Randomstate_10_Out.csv',
    'Randomstate_20_Out.csv',
    'Randomstate_30_Out.csv',
    'Randomstate_40_Out.csv',
    'Randomstate_50_Out.csv',
    'Randomstate_60_Out.csv',
    'Randomstate_70_Out.csv',
    'Randomstate_80_Out.csv',
    'Randomstate_90_Out.csv',
    'ReverseCurbstone_5_Out.csv',
    'Speedbump_5_Out.csv',
    # CRASH
    'ISO13232-1_Out.csv',
    'ISO13232-2_Out.csv',
    'ISO13232-3_Out.csv',
    'ISO13232-4_Out.csv',
    'ISO13232-5_Out.csv',
    'ISO13232-6_Out.csv',
    'ISO13232-7_Out.csv',
    'Iso_115_Out.csv',
    'Iso_131_Out.csv',
    'Iso_132_Out.csv',
    'Iso_226_3_Out.csv',
    'Iso_241_Out.csv',
    'Iso_242_2_Out.csv',
    'Iso_243_Out.csv',
    'Iso_312_2_Out.csv',
    'Iso_313_2_Out.csv',
    'Iso_314_2_Out.csv',
    'Iso_512_2_Out.csv',
    'Iso_513_2_Out.csv',
    'Iso_514_2_Out.csv',
    'Iso_623_Out.csv',
    'Iso_624_Out.csv',
    'Iso_648_Out.csv',
    'Iso_711_Out.csv',
    'Iso_712_Out.csv'
]

scenario_id = 0
scenario_data = pd.read_csv(os.path.join(datapath, 'ControlScenarios', test_paths[scenario_id]))
# input data consists of selected features
X_scenario = scenario_data[features].to_numpy()
# output data consists of crash label
y_scenario = scenario_data[['CrashLabelML']].to_numpy()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from local_pkg.msg import Local
import threading
import os
import sys
import math
from math import sqrt, atan2, sin, atan, cos, sqrt
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry,Path
from tf.transformations import euler_from_quaternion
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus, EventInfo
import numpy as np
import matplotlib.pyplot as plt
import json
import serial
from enum import Enum
from morai_msgs.srv import MoraiEventCmdSrv
# import optimal_franet_morai

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from . import draw
# from . import cubic_spline as cs
import rospy
import time

# from .Path import PATH
# from .VehicleModel import Node
# from .VehicleParameter import Parameter

class Gear(Enum):
    P = 1
    R = 2
    N = 3
    D = 4

class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[0 : -1]]
        dy = [node.y - y for y in self.cy[0 : -1]]
        dist = np.hypot(dx, dy)

        self.ind_old = int(np.argmin(dist))

        ind = self.ind_old

        return ind

class Parameter:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi] * phi = yaw angle
    NU = 2  # input vector: u = [acceleration, steer]
    T = 20  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs 

    dist_stop = 5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    iter_max = 5  # max iteration
    target_speed = 20.0 * 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 0.1  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 0.87  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.49  # [m] distance from rear to vehicle back end of vehicle
    W = 0.90  # [m] width of vehicle
    WD = 0.965 * W  # [m] distance between left-right wheels
    WB = 3  # [m] Wheel base
    TR = 0.65  # [m] Tyre radius
    TW = 0.17  # [m] Tyre width
    
    steer_max = np.deg2rad(40.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(35.0)  # maximum steering speed [rad/s]
    speed_max = 20.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]

class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        # ROS Publish
        # rospy.init_node("Serial_IO", anonymous=False)
        # self.serial_pub = rospy.Publisher("/serial", Serial_Info, queue_size=1)
        # Messages/Data
        # self.serial_msg = Serial_Info()  # Message o publish

        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = 1
        # if self.serial_msg.gear == 0:
        #     self.direct = 1 # 전진
        # elif self.serial_msg.gear == 2:
        #     self.direct = -1 # 후진

    def update(self, a, delta, gear): #가속도랑 조향각, 방향
        delta = self.limit_input_delta(delta) #최대 조향각 조절
        self.x += self.v * math.cos(self.yaw) * Parameter.dt 
        self.y += self.v * math.sin(self.yaw) * Parameter.dt
        self.yaw += self.v / Parameter.WB * math.tan(delta) * Parameter.dt
        if gear == 0:
            self.direct = 1
        elif gear == 2:
            self.direct = -1    
        self.v += self.direct * a * Parameter.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= Parameter.steer_max:
            return Parameter.steer_max

        if delta <= -Parameter.steer_max:
            return -Parameter.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= Parameter.speed_max:
            return Parameter.speed_max

        if v <= Parameter.speed_min:
            return Parameter.speed_min

        return v
    

def calc_ref_trajectory_in_T_step(node, ind, ref_path, sp):
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory
    """
    z_ref = np.zeros((Parameter.NX, Parameter.T + 1))
    length = ref_path.length

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = ref_path.cyaw[ind]
    z_ref[3, 0] = sp[ind]

    dist_move = 0.0

    for i in range(1, Parameter.T + 1):
        dist_move += node.v * Parameter.dt
        ind_move = int(round(dist_move / Parameter.d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = ref_path.cyaw[index]
        z_ref[3, i] = sp[index]

    return z_ref


def predict_states_in_T_step(z0, a, delta, z_ref):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: initial state
    :param a: acceleration strategy of last time
    :param delta: delta strategy of last time
    :param z_ref: reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(Parameter.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, Parameter.T + 1)):
        node.update(ai, di, node.direct)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar    
def normalize_angle_degrees(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
def mpc_pure_pursuit(node, ego_ind, global_path_x, global_path_y, mission):
    # i = 0 # v*dt = 거리 >> 내 x + 거리 = > 인덱스 찾아서 그 때의 스티어를 구하자
    # steer_list = []
    # for i in range(Parameter.T + 1):
    #     i += 30
    #     if gear == 0:
    #         target_index = ego_ind + i
    #     elif gear == 2:
    #         target_index = ego_ind - i

    #     target_x, target_y = global_path_x[target_index], global_path_y[target_index]
    #     tmp = (math.degrees(math.atan2(target_y - node.y, target_x - node.x))) % 360

    #     # Convert node_yaw from radians to degrees
    #     node_yaw_degrees = np.degrees(node.yaw)
    
    #     # Normalize desired_yaw to -180 to 180 range
    #     desired_yaw = normalize_angle_degrees(tmp)

    #     # print("node_yaw:", node_yaw_degrees)
    #     # print("desired_yaw:", desired_yaw)
    #     alpha = (desired_yaw - node_yaw_degrees) % 360
    #     alpha = normalize_angle_degrees(alpha)  # Ensure alpha is in the -180 to 180 range
    #     # print("alpha: ", alpha)
    #     angle = math.atan2(2.0 * Parameter.WB * math.sin(np.radians(alpha)), 1)
    #     # print('angle:', angle)
    
    #     # if angle < 0.5 and angle > -0.5:ㄴ
    #     #     angle = 0
        
    #     # if abs(angle) > 5: 
    #     #     angle *= 0.7

    #     steer_list.append(angle)
    i = 0 # v*dt = 거리 >> 내 x + 거리 = > 인덱스 찾아서 그 때의 스티어를 구하자
    steer_list = []
    for i in range(Parameter.T + 1):

        if mission:
            i += 40
        else:
            i += 47
        
        target_index = ego_ind + i
    

        target_x, target_y = global_path_x[target_index], global_path_y[target_index]
        tmp = (math.degrees(math.atan2(target_y - node.y, target_x - node.x))) % 360

        # Convert node_yaw from radians to degrees
        node_yaw_degrees = np.degrees(node.yaw)
    
        # Normalize desired_yaw to -180 to 180 range
        desired_yaw = normalize_angle_degrees(tmp)

        # print("node_yaw:", node_yaw_degrees)
        # print("desired_yaw:", desired_yaw)
        alpha = (desired_yaw - node_yaw_degrees) % 360
        alpha = normalize_angle_degrees(alpha)  # Ensure alpha is in the -180 to 180 range
        # print("alpha: ", alpha)
        angle = math.atan2(2.0 * Parameter.WB * math.sin(np.radians(alpha)), 1)
        # print('angle:', angle)
    
        # if angle < 0.5 and angle > -0.5:
        #     angle = 0
        
        # if abs(angle) > 5: 
        #     angle *= 0.7

        steer_list.append(angle/3)
        
    return steer_list
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle
def calc_speed_profile(cx, cy, cyaw, target_speed):
    """# def localcallback(self, msg):
    #     self.ego_info.x = msg.x
    #     self.ego_info.y = msg.y
    #     self.ego_info.heading = msg.heading
    #     self.ego_info.speeed = msg.speeed
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed/3.6] * len(cx)
    direction = 1.0  # forward
    speed_profile[-1] = 0.0

    return speed_profile

def mpc_predict_next_state(z_ref, x0, y0, yaw0, v0, v_pre, steer_list, gear):

    future_list = z_ref * 0.0

    future_list[0, 0] = x0
    future_list[1, 0] = y0
    future_list[2, 0] = yaw0
    future_list[3, 0] = v0
    # a_list = []

    v_pre = None

    for i in range(Parameter.T):
        x1 = x0 + v0 * math.cos(yaw0) * Parameter.dt 
        y1 = y0 + v0 * math.sin(yaw0) * Parameter.dt
        yaw1 = yaw0 + v0 / Parameter.WB * math.tan(steer_list[i]) * Parameter.dt
        if gear == 0:
            direct = 1
        elif gear == 2:
            direct = -1
        
        if v_pre == None:
            # v_pre = v0
            a = 10
            v1 = v0 + direct * a * Parameter.dt
        else:
            a = (v0 - v_pre)/Parameter.dt
            v1 = v0 + direct * a * Parameter.dt
        
        future_list[0, i+1] = x1
        future_list[1, i+1] = y1
        future_list[2, i+1] = yaw1
        future_list[3, i+1] = v1
        # a_list.append(a)

        x0 = x1
        y0 = y1
        yaw0 = yaw1
        v_pre = v0
        v0 = v1

    # return future_list, a_list
    return future_list

def mpc_cost_function(z_ref, z_bar, steer_list):
    i = 1
    cost = []
    for i in range(len(z_ref)):
        cost_function_1 = (z_ref[0][i] - z_bar[0][i])**2
        cost_function_2 = (z_ref[1][i] - z_bar[1][i])**2
        cost_function_3 = (z_ref[2][i] - z_bar[2][i])**2
        cost_function_4 = (z_ref[3][i] - z_bar[3][i])**2
        cost_function = cost_function_1 + cost_function_2 + cost_function_3 + cost_function_4
        cost.append(cost_function)

    selected_index = int(np.argmin(cost))
    return selected_index


class MPC:
    def __init__(self,ego_info = 0,shared = 0,plan= 0,parking = 0):
        self.k = 0.55
        # self.k = 0.15
        self.WB = 3.000 # wheel base
        self.current_postion = Point()
        self.ck = 0
        self.vehicle_yaw = 0
        self.current_vel = 0
        self.flag = 0
        self.Parking_mission = False
        self.gear_change = False
        self.gear_cmd_sent = False
        self.count = 0
        # Shark MORAI
        rospy.init_node("MPC", anonymous=True)
        rospy.Subscriber("odom",Odometry,self.odom_callback)
        # rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.wait_for_service('/Service_MoraiEventCmd')
        self.event_cmd_srv = rospy.ServiceProxy('Service_MoraiEventCmd',MoraiEventCmdSrv)
        # self.gear_cmd = EventInfo()
        # self.gear_cmd.option = 3
        # self.gear_cmd.ctrl_mode = 3
        # self.gear_cmd.gear = 4
        # self.gear_cmd_resp = self.event_cmd_srv(self.gear_cmd)
        self.send_gear_cmd(Gear.D.value)
        
        
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg= CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType=2
        
        # Shark MORAI
        self.ego_info = self.current_postion
        print(self.ego_info)
        # self.ego_info.heading=self.vehicle_yaw
        # self.ego_info.speeed = self.current_vel
        
        #####!!! Ego의 x,y,heading,speed 정보 받아오기

        # Ego information      
        self.curPosition_x = []
        self.curPosition_y = []
        # error
        self.error = []
        self.abs_error = []

        # # rviz
        # self.global_path_pub = rospy.Publisher('/global_path', customPath, queue_size = 1)
        # self.trajectory_pub = rospy.Publisher('/trajectory', customPath, queue_size = 1)

        # rospy Rate (Hz)
        self.rt = 50
        ####### 글로벌 path 받아오기
        
        self.global_path = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/seongnam_final_global_path.json")  
        self.global_path = list(self.global_path)
        self.cx = self.global_path[0]
        self.cy = self.global_path[1]
        self.cyaw = self.global_path[2]
        print("length",len(self.cx),"self.cx:",self.cx[0],"self.cy",self.cy[0])
        self.local_path = None
        # MPC initial
        self.sp = calc_speed_profile(self.cx, self.cy, self.cyaw, Parameter.target_speed) # speed profile
        
        # self.ref_path = PATH(self.cx, self.cy, self.cyaw, self.ck)
        
        self.node = Node(x=self.cx[0], y=self.cy[0], yaw=self.cyaw[0], v=0.0)

        self.time = 0.0
        self.x = [self.node.x]
        self.y = [self.node.y]
        self.yaw = [self.node.yaw]
        self.v = [self.node.v]
        self.t = []
        self.prev_x = self.cx[0]
        self.prev_y = self.cy[0]

        # state
        # self.state = "driving"
 
    def update_flag(self, ego_ind):
        if 3500 <= ego_ind <= 4200:
            if not self.Parking_mission and abs(ego_ind - 3756) < 20 and self.count == 0:
                self.flag = 0
                self.Parking_mission = True
                self.count += 1
            else:
                self.flag = 1
        else:
            self.flag = 0
        
        # if self.Parking_mission:
        #     if ego_ind < 3500 or ego_ind >= 4200:
        #         self.Parking_mission = False
        #         self.flag = 0


    def run(self):
        rate = rospy.Rate(self.rt)
        ##########################
        
        while not rospy.is_shutdown():
                    
            
        
            ###################################################################
            self.time += 1/self.rt 
            # time = 1/(rospy.rate[Hz]) = [s] 

            self.node.x = self.ego_info.x
            self.node.y = self.ego_info.y
            self.node.yaw = self.vehicle_yaw
            self.node.v = self.current_vel
            #####################################################################
            

            
            print("선속도!선속도!:", self.node.v)
            
            print("self.node.x",self.node.x,"self.node.y",self.node.y)
            ego_ind = self.nearest_index(self.node)
            print("current index:", ego_ind)
            self.update_flag(ego_ind)
            print("flag:", self.flag)
            print("mission:",self.Parking_mission)
            # if 1563 <= ego_ind <= 2041:
            #     self.local_path = optimal_franet_morai()
            #     self.ref_path = PATH(self.local_path.x, self.local_path.y, self.local_path.yaw, self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # else:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            
            # self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            # input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            # input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            
            self.parking_path = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/parking_data.json")
            if self.flag == 0 and self.Parking_mission == False:
                self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
                input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            elif self.flag == 1 and self.Parking_mission == False:
                print("ego_ind:", ego_ind)
                self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
                input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
                input_speed = 15
            elif self.flag == 0 and self.Parking_mission == True:
                if not self.gear_cmd_sent:
                    self.send_gear_cmd(Gear.R.value)
                    self.gear_cmd_sent = True

               
               


                print("dlsadllsadlsaldlsadlsadlsadlsadlsadl")

                
                


                if self.vehicle_yaw >= 1.52:
                    input_speed = 2
                    input_steer = 0
                else:
                    self.cx = self.parking_path[0]
                    self.cy = self.parking_path[1]
                    self.cyaw = self.parking_path[2]
                    self.ref_path = PATH(self.parking_path[0],self.parking_path[1],self.parking_path[2],self.ck)
                    parking_ego_ind = self.nearest_index(self.node)
                    print(parking_ego_ind)
                    input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(parking_ego_ind)
                    input_speed = 2
                if self.node.y <= 105.95:
                    self.send_gear_cmd(Gear.D.value)
                    self.cx = self.global_path[0]
                    self.cy = self.global_path[1]
                    self.cyaw = self.global_path[2]
                    self.Parking_mission = False

                print(self.vehicle_yaw)

            # if 3500 <= ego_ind <= 4200:
            #     self.flag = 1
            #     if ego_ind >= 3756:
            #         self.flag = 2
            #         print("Parking Start")
            # if self.flag == 0:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # elif self.flag == 1:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # elif self.flag == 2:
            #     self.ref_path = PATH(self.parking_path[0],self.parking_path[1],self.parking_path[2],self.ck)
            #     ego_ind = self.nearest_index(self.node)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            #     input_speed = 3
            


            # print("                  x         y        yaw       v")
            # for i in range(Parameter.T):
            #     print("Predict {:02d}: ".format(i+1),end='   ',)
            #     print('{:.4f}'.format(z_ref[0,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[1,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[2,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[3,i]))
            # print("selected future x: ", z_ref[0, selected_index])
            # print("selected future y: ", z_ref[1, selected_index])
            # print("selected future yaw: ", z_ref[2, selected_index])
            # print("selected future v: ", z_ref[3, selected_index])
            # print("current indexself.vehicle_yaw: ", ego_ind)
            # print("state: ", self.state)
            # print("gear: ", self.serial_msg.gear)
            self.ctrl_cmd_msg.steering = input_steer
            self.ctrl_cmd_msg.velocity = input_speed
            print("steer: ", self.ctrl_cmd_msg.steering)
            print("speed: ", self.ctrl_cmd_msg.velocity)

            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            rate.sleep()
            # return input_speed,input_steer,input_break
    

    def send_gear_cmd(self,gear_mode):
        print("Hi im send_gear")
        while (abs(self.current_vel) > 0.1):
            self.ctrl_cmd_msg.velocity = 0
            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
        gear_cmd = EventInfo()
        gear_cmd.option = 3
        gear_cmd.ctrl_mode = 3
        gear_cmd.gear = gear_mode
        gear_cmd_resp = self.event_cmd_srv(gear_cmd)
        rospy.loginfo(gear_cmd)
    def save_position(self):
        self.curPosition_x.append(self.ego_info.x)
        self.curPosition_y.append(self.ego_info.y)

    
    # def setValue(self, speed, steer, brake):
    #     self.control_input.speed = speed
    #     self.control_input.steer = steer
    #     self.control_input.brake = brake

    def Model_Predictive_Control(self, ego_ind):
        self.x.append(self.node.x)
        self.y.append(self.node.y)
        self.yaw.append(self.node.yaw)
        self.v.append(self.node.v)
        self.t.append(time)
        
        z_ref = calc_ref_trajectory_in_T_step(self.node, ego_ind, self.ref_path, self.sp) # 내 인덱스에서 미래 reference state
    
        steer_list = mpc_pure_pursuit(self.node, ego_ind, self.cx, self.cy, self.Parking_mission) # 내 위치에서 뽑아낸 미래 예측 steer들
        # print(steer_list)
        z_bar = mpc_predict_next_state(z_ref, self.node.x, self.node.y, self.node.yaw, self.node.v, self.v[-1], steer_list, 0) # 미래 예측된 state # 위에서 뽑은 걸 기반으로 뽑아낸 예측 state
        # print(steer_list)
        selected_index = mpc_cost_function(z_ref, z_bar, steer_list) # z_ref와 z_bar을 기반으로 해서 다음 state index 정하기
        # print(steer_list)
        self.node.update(10, steer_list[selected_index], 0)

        return z_ref[3, selected_index], steer_list[selected_index], 0, z_ref, selected_index
        
    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """
        print("self.cx:",self.cx[0],"node.x:",node.x)
        print("self.cy:",self.cy[0],"node.y:",node.y)
        dx = [node.x - x for x in self.cx[0 : -1]]
        dy = [node.y - y for y in self.cy[0 : -1]]
        # dx = [node.x - node.x]
        # dy = [node.y - node.y]
        # dx = [x for x in self.cx[0 : -1] - (x for x in self.cx[0 : -1])]
        # dy = [y for y in self.cy[0 : -1] - (y for y in self.cy[0 : -1])]
        dist = np.hypot(dx, dy)

        self.ind_old = int(np.argmin(dist))

        ind = self.ind_old

        return ind
    
    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y
        ################################################################### 선속도 계산
        dt = 0.2

        
        delta_x = self.current_postion.x - self.prev_x
        delta_y = self.current_postion.y - self.prev_y
        distance = math.sqrt(delta_x**2 + delta_y**2)
        self.current_vel = distance /dt  * 3.6

        self.prev_x = self.current_postion.x
        self.prev_y = self.current_postion.y
      

        # self.node.v = self.current_vel
            

       

        # print(self.current_postion.x,self.current_postion.y, self.vehicle_yaw)
    
    # def status_callback(self, msg):
    #     self.is_status = True
    #     self.current_vel = msg.velocity.x
    
    def load_global_path(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        global_path_x = []
        global_path_y = []
        global_path_yaw = []

        for key in sorted(data.keys(), key=int):
            point = data[key]
            global_path_x.append(point[0])
            global_path_y.append(point[1])
            global_path_yaw.append(point[2])

        return global_path_x, global_path_y, global_path_yaw
            
if __name__ == "__main__":
    morai = MPC().run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from local_pkg.msg import Local
import threading
import os
import sys
import math
from math import sqrt, atan2, sin, atan, cos, sqrt
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry,Path
from tf.transformations import euler_from_quaternion
from morai_msgs.msg import CtrlCmd, EgoVehicleStatus, EventInfo
import numpy as np
import matplotlib.pyplot as plt
import json
import serial
from enum import Enum
from morai_msgs.srv import MoraiEventCmdSrv
# import optimal_franet_morai

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from . import draw
# from . import cubic_spline as cs
import rospy
import time

# from .Path import PATH
# from .VehicleModel import Node
# from .VehicleParameter import Parameter

class Gear(Enum):
    P = 1
    R = 2
    N = 3
    D = 4

class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[0 : -1]]
        dy = [node.y - y for y in self.cy[0 : -1]]
        dist = np.hypot(dx, dy)

        self.ind_old = int(np.argmin(dist))

        ind = self.ind_old

        return ind

class Parameter:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi] * phi = yaw angle
    NU = 2  # input vector: u = [acceleration, steer]
    T = 20  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs 

    dist_stop = 5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    iter_max = 5  # max iteration
    target_speed = 20.0 * 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 0.1  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 0.87  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.49  # [m] distance from rear to vehicle back end of vehicle
    W = 0.90  # [m] width of vehicle
    WD = 0.965 * W  # [m] distance between left-right wheels
    WB = 3  # [m] Wheel base
    TR = 0.65  # [m] Tyre radius
    TW = 0.17  # [m] Tyre width
    
    steer_max = np.deg2rad(40.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(35.0)  # maximum steering speed [rad/s]
    speed_max = 20.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]

class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        # ROS Publish
        # rospy.init_node("Serial_IO", anonymous=False)
        # self.serial_pub = rospy.Publisher("/serial", Serial_Info, queue_size=1)
        # Messages/Data
        # self.serial_msg = Serial_Info()  # Message o publish

        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = 1
        # if self.serial_msg.gear == 0:
        #     self.direct = 1 # 전진
        # elif self.serial_msg.gear == 2:
        #     self.direct = -1 # 후진

    def update(self, a, delta, gear): #가속도랑 조향각, 방향
        delta = self.limit_input_delta(delta) #최대 조향각 조절
        self.x += self.v * math.cos(self.yaw) * Parameter.dt 
        self.y += self.v * math.sin(self.yaw) * Parameter.dt
        self.yaw += self.v / Parameter.WB * math.tan(delta) * Parameter.dt
        if gear == 0:
            self.direct = 1
        elif gear == 2:
            self.direct = -1    
        self.v += self.direct * a * Parameter.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= Parameter.steer_max:
            return Parameter.steer_max

        if delta <= -Parameter.steer_max:
            return -Parameter.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= Parameter.speed_max:
            return Parameter.speed_max

        if v <= Parameter.speed_min:
            return Parameter.speed_min

        return v
    

def calc_ref_trajectory_in_T_step(node, ind, ref_path, sp):
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory
    """
    z_ref = np.zeros((Parameter.NX, Parameter.T + 1))
    length = ref_path.length

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = ref_path.cyaw[ind]
    z_ref[3, 0] = sp[ind]

    dist_move = 0.0

    for i in range(1, Parameter.T + 1):
        dist_move += node.v * Parameter.dt
        ind_move = int(round(dist_move / Parameter.d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = ref_path.cyaw[index]
        z_ref[3, i] = sp[index]

    return z_ref


def predict_states_in_T_step(z0, a, delta, z_ref):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: initial state
    :param a: acceleration strategy of last time
    :param delta: delta strategy of last time
    :param z_ref: reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(Parameter.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, Parameter.T + 1)):
        node.update(ai, di, node.direct)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar    
def normalize_angle_degrees(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
def mpc_pure_pursuit(node, ego_ind, global_path_x, global_path_y, mission):
    # i = 0 # v*dt = 거리 >> 내 x + 거리 = > 인덱스 찾아서 그 때의 스티어를 구하자
    # steer_list = []
    # for i in range(Parameter.T + 1):
    #     i += 30
    #     if gear == 0:
    #         target_index = ego_ind + i
    #     elif gear == 2:
    #         target_index = ego_ind - i

    #     target_x, target_y = global_path_x[target_index], global_path_y[target_index]
    #     tmp = (math.degrees(math.atan2(target_y - node.y, target_x - node.x))) % 360

    #     # Convert node_yaw from radians to degrees
    #     node_yaw_degrees = np.degrees(node.yaw)
    
    #     # Normalize desired_yaw to -180 to 180 range
    #     desired_yaw = normalize_angle_degrees(tmp)

    #     # print("node_yaw:", node_yaw_degrees)
    #     # print("desired_yaw:", desired_yaw)
    #     alpha = (desired_yaw - node_yaw_degrees) % 360
    #     alpha = normalize_angle_degrees(alpha)  # Ensure alpha is in the -180 to 180 range
    #     # print("alpha: ", alpha)
    #     angle = math.atan2(2.0 * Parameter.WB * math.sin(np.radians(alpha)), 1)
    #     # print('angle:', angle)
    
    #     # if angle < 0.5 and angle > -0.5:ㄴ
    #     #     angle = 0
        
    #     # if abs(angle) > 5: 
    #     #     angle *= 0.7

    #     steer_list.append(angle)
    i = 0 # v*dt = 거리 >> 내 x + 거리 = > 인덱스 찾아서 그 때의 스티어를 구하자
    steer_list = []
    for i in range(Parameter.T + 1):

        if mission:
            i += 25
        else:
            i += 47

        print("ego_ind:", ego_ind)
    
        
        target_index = ego_ind + i
        print("target_ind:",target_index)
    

        target_x, target_y = global_path_x[target_index], global_path_y[target_index]
        tmp = (math.degrees(math.atan2(target_y - node.y, target_x - node.x))) % 360

        # Convert node_yaw from radians to degrees
        node_yaw_degrees = np.degrees(node.yaw)
    
        # Normalize desired_yaw to -180 to 180 range
        desired_yaw = normalize_angle_degrees(tmp)

        # print("node_yaw:", node_yaw_degrees)
        # print("desired_yaw:", desired_yaw)
        alpha = (desired_yaw - node_yaw_degrees) % 360
        alpha = normalize_angle_degrees(alpha)  # Ensure alpha is in the -180 to 180 range
        # print("alpha: ", alpha)
        angle = math.atan2(2.0 * Parameter.WB * math.sin(np.radians(alpha)), 1)
        # print('angle:', angle)
    
        # if angle < 0.5 and angle > -0.5:
        #     angle = 0
        
        # if abs(angle) > 5: 
        #     angle *= 0.7
        # print(angle)
        steer_list.append(angle/3)
        
    return steer_list
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle
def calc_speed_profile(cx, cy, cyaw, target_speed):
    """# def localcallback(self, msg):
    #     self.ego_info.x = msg.x
    #     self.ego_info.y = msg.y
    #     self.ego_info.heading = msg.heading
    #     self.ego_info.speeed = msg.speeed
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed/3.6] * len(cx)
    direction = 1.0  # forward
    speed_profile[-1] = 0.0

    return speed_profile

def mpc_predict_next_state(z_ref, x0, y0, yaw0, v0, v_pre, steer_list, gear):

    future_list = z_ref * 0.0

    future_list[0, 0] = x0
    future_list[1, 0] = y0
    future_list[2, 0] = yaw0
    future_list[3, 0] = v0
    # a_list = []

    v_pre = None

    for i in range(Parameter.T):
        x1 = x0 + v0 * math.cos(yaw0) * Parameter.dt 
        y1 = y0 + v0 * math.sin(yaw0) * Parameter.dt
        yaw1 = yaw0 + v0 / Parameter.WB * math.tan(steer_list[i]) * Parameter.dt
        if gear == 0:
            direct = 1
        elif gear == 2:
            direct = -1
        
        if v_pre == None:
            # v_pre = v0
            a = 10
            v1 = v0 + direct * a * Parameter.dt
        else:
            a = (v0 - v_pre)/Parameter.dt
            v1 = v0 + direct * a * Parameter.dt
        
        future_list[0, i+1] = x1
        future_list[1, i+1] = y1
        future_list[2, i+1] = yaw1
        future_list[3, i+1] = v1
        # a_list.append(a)

        x0 = x1
        y0 = y1
        yaw0 = yaw1
        v_pre = v0
        v0 = v1

    # return future_list, a_list
    return future_list

def mpc_cost_function(z_ref, z_bar, steer_list):
    i = 1
    cost = []
    for i in range(len(z_ref)):
        cost_function_1 = (z_ref[0][i] - z_bar[0][i])**2
        cost_function_2 = (z_ref[1][i] - z_bar[1][i])**2
        cost_function_3 = (z_ref[2][i] - z_bar[2][i])**2
        cost_function_4 = (z_ref[3][i] - z_bar[3][i])**2
        cost_function = cost_function_1 + cost_function_2 + cost_function_3 + cost_function_4
        cost.append(cost_function)

    selected_index = int(np.argmin(cost))
    return selected_index


class MPC:
    def __init__(self,ego_info = 0,shared = 0,plan= 0,parking = 0):
        self.k = 0.55
        # self.k = 0.15
        self.WB = 3.000 # wheel base
        self.current_postion = Point()
        self.ck = 0
        self.vehicle_yaw = 0
        self.current_vel = 0
        self.flag = 0
        self.Parking_mission = False
        self.Parking_mission2 = False
        self.gear_change = False
        self.gear_cmd_sent = False
        self.parking_ego_ind = 0
        self.ret = False
        self.count = 0
        # Shark MORAI
        rospy.init_node("MPC", anonymous=True)
        rospy.Subscriber("odom",Odometry,self.odom_callback)
        # rospy.Subscriber("Ego_topic", EgoVehicleStatus, self.status_callback)
        rospy.wait_for_service('/Service_MoraiEventCmd')
        self.event_cmd_srv = rospy.ServiceProxy('Service_MoraiEventCmd',MoraiEventCmdSrv)
        # self.gear_cmd = EventInfo()
        # self.gear_cmd.option = 3
        # self.gear_cmd.ctrl_mode = 3
        # self.gear_cmd.gear = 4
        # self.gear_cmd_resp = self.event_cmd_srv(self.gear_cmd)
        self.send_gear_cmd(Gear.D.value)
        
        
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg= CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType=2
        
        # Shark MORAI
        self.ego_info = self.current_postion
        print(self.ego_info)
        # self.ego_info.heading=self.vehicle_yaw
        # self.ego_info.speeed = self.current_vel
        
        #####!!! Ego의 x,y,heading,speed 정보 받아오기

        # Ego information      
        self.curPosition_x = []
        self.curPosition_y = []
        # error
        self.error = []
        self.abs_error = []

        # # rviz
        # self.global_path_pub = rospy.Publisher('/global_path', customPath, queue_size = 1)
        # self.trajectory_pub = rospy.Publisher('/trajectory', customPath, queue_size = 1)

        # rospy Rate (Hz)
        self.rt = 50
        ####### 글로벌 path 받아오기
        
        self.global_path = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/seongnam_final_global_path.json")  
        self.global_path = list(self.global_path)
        self.cx = self.global_path[0]
        self.cy = self.global_path[1]
        self.cyaw = self.global_path[2]
        print("length",len(self.cx),"self.cx:",self.cx[0],"self.cy",self.cy[0])
        self.local_path = None
        # MPC initial
        self.sp = calc_speed_profile(self.cx, self.cy, self.cyaw, Parameter.target_speed) # speed profile
        
        # self.ref_path = PATH(self.cx, self.cy, self.cyaw, self.ck)
        
        self.node = Node(x=self.cx[0], y=self.cy[0], yaw=self.cyaw[0], v=0.0)

        self.time = 0.0
        self.x = [self.node.x]
        self.y = [self.node.y]
        self.yaw = [self.node.yaw]
        self.v = [self.node.v]
        self.t = []
        self.prev_x = self.cx[0]
        self.prev_y = self.cy[0]

        # state
        # self.state = "driving"
 
    def update_flag(self, ego_ind):
        if 3314 <= ego_ind <= 4200:
            if not self.Parking_mission and abs(ego_ind - 3314) < 20 and self.count == 0:
                self.flag = 0
                self.Parking_mission = True
                self.count += 1

            else:
                self.flag = 1
        else:
            self.flag = 0
        
        # if self.Parking_mission:
        #     if ego_ind < 3500 or ego_ind >= 4200:
        #         self.Parking_mission = False
        #         self.flag = 0


    def run(self):
        rate = rospy.Rate(self.rt)
        ##########################
        
        while not rospy.is_shutdown():
                    
            
        
            ###################################################################
            self.time += 1/self.rt 
            # time = 1/(rospy.rate[Hz]) = [s] 

            self.node.x = self.ego_info.x
            self.node.y = self.ego_info.y
            self.node.yaw = self.vehicle_yaw
            self.node.v = self.current_vel
            #####################################################################
            

            
            print("선속도!선속도!:", self.node.v)
            
            print("self.node.x",self.node.x,"self.node.y",self.node.y)
            ego_ind = self.nearest_index(self.node)
            print("current index:", ego_ind)
            self.update_flag(ego_ind)
            print("flag:", self.flag)
            print("mission:",self.Parking_mission)
            # if 1563 <= ego_ind <= 2041:
            #     self.local_path = optimal_franet_morai()
            #     self.ref_path = PATH(self.local_path.x, self.local_path.y, self.local_path.yaw, self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # else:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            
            # self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            # input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            # input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            
            self.parking_path = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/parking_data.json")
            self.parking_path2 = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/parking_data2.json")
            self.local_path = self.load_global_path("/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/optimal_map.json")
            if self.flag == 0 and self.Parking_mission == False and self.Parking_mission2 == False:
                self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
                input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            elif self.flag == 1 and self.Parking_mission == False and self.Parking_mission2 == False:
                print("ego_ind:", ego_ind)
                self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
                input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
                input_speed = 15
            elif self.flag == 0 and self.Parking_mission == True:
                print("Parkinging~~~~~~~~~~~")
                self.cx = self.parking_path[0]
                self.cy = self.parking_path[1]
                self.cyaw = self.parking_path[2]
                self.ref_path = PATH(self.parking_path[0],self.parking_path[1],self.parking_path[2],self.ck)
                self.parking_ego_ind = self.nearest_index(self.node)
                print(self.parking_ego_ind)
                input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(self.parking_ego_ind)
                input_speed = 3
                if self.Parking_mission and self.count != 0 and 400 <= abs(self.parking_ego_ind) <= 404:
                    print("ASdsadsadsadasdasdsadsadsad")
                    self.Parking_mission2 = True
                    self.Parking_mission = False
                    self.flag = 0
            elif self.flag == 0 and self.Parking_mission2 == True:
                print("backward gogogogogo")
                print("Parking_mission2:", self.Parking_mission2)
                if not self.gear_cmd_sent:
                    self.send_gear_cmd(Gear.R.value)
                    self.gear_cmd_sent = True
                # self.cx = self.parking_path2[0]
                # self.cy = self.parking_path2[1]
                # self.cyaw = self.parking_path2[2]
                # self.ref_path = PATH(self.parking_path2[0],self.parking_path2[1],self.parking_path2[2],1)
                # print(self.node.x,self.node.y)
                # parking_ego_ind = self.nearest_index(self.node)
                # print(parking_ego_ind)
                # input_speed,input_steer,input_break,z_ref,selected_index = self.Model_Predictive_Control(parking_ego_ind)
                # input_speed = 1

                input_speed = 1
                input_steer = -0.4


                if self.vehicle_yaw >=1.58:
                    input_speed = 2
                    input_steer = 0

                if self.node.y <= 105.95:
                    self.send_gear_cmd(Gear.D.value)
                    self.ret = True
                    # self.cx = self.global_path[0]
                    # self.cy = self.global_path[1]
                    # self.cyaw = self.global_path[2]
                    # self.Parking_mission2 = False
                if self.node.y >= (105.9 + 5) and self.ret:
                    self.cx = self.global_path[0]
                    self.cy = self.global_path[1]
                    self.cyaw = self.global_path[2]
                    # self.ref_path = PATH(self.parking_path2[0],self.parking_path2[1],self.parking_path2[2],1)
                    # print(self.node.x,self.node.y)
                    # parking_ego_ind = self.nearest_index(self.node)
                    # print(parking_ego_ind)
                    # input_speed,input_steer,input_break,z_ref,selected_index = self.Model_Predictive_Control(parking_ego_ind)
                    self.Parking_mission2 = False
    


                print(self.vehicle_yaw)

            # if 3500 <= ego_ind <= 4200:
            #     self.flag = 1
            #     if ego_ind >= 3756:
            #         self.flag = 2
            #         print("Parking Start")
            # if self.flag == 0:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # elif self.flag == 1:
            #     self.ref_path = PATH(self.global_path[0], self.global_path[1], self.global_path[2], self.ck)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            # elif self.flag == 2:
            #     self.ref_path = PATH(self.parking_path[0],self.parking_path[1],self.parking_path[2],self.ck)
            #     ego_ind = self.nearest_index(self.node)
            #     input_speed,input_steer,input_break, z_ref, selected_index = self.Model_Predictive_Control(ego_ind)
            #     input_speed = 3
            


            # print("                  x         y        yaw       v")
            # for i in range(Parameter.T):
            #     print("Predict {:02d}: ".format(i+1),end='   ',)
            #     print('{:.4f}'.format(z_ref[0,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[1,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[2,i]),end='   ',)
            #     print('{:.4f}'.format(z_ref[3,i]))
            # print("selected future x: ", z_ref[0, selected_index])
            # print("selected future y: ", z_ref[1, selected_index])
            # print("selected future yaw: ", z_ref[2, selected_index])
            # print("selected future v: ", z_ref[3, selected_index])
            # print("current indexself.vehicle_yaw: ", ego_ind)
            # print("state: ", self.state)
            # print("gear: ", self.serial_msg.gear)
            self.ctrl_cmd_msg.steering = input_steer
            self.ctrl_cmd_msg.velocity = input_speed
            print("steer: ", self.ctrl_cmd_msg.steering)
            print("speed: ", self.ctrl_cmd_msg.velocity)

            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

            rate.sleep()
            # return input_speed,input_steer,input_break
    

    def send_gear_cmd(self,gear_mode):
        print("Hi im send_gear")
        while (abs(self.current_vel) > 0.1):
            self.ctrl_cmd_msg.velocity = 0
            self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
        gear_cmd = EventInfo()
        gear_cmd.option = 3
        gear_cmd.ctrl_mode = 3
        gear_cmd.gear = gear_mode
        gear_cmd_resp = self.event_cmd_srv(gear_cmd)
        rospy.loginfo(gear_cmd)
    def save_position(self):
        self.curPosition_x.append(self.ego_info.x)
        self.curPosition_y.append(self.ego_info.y)

    
    # def setValue(self, speed, steer, brake):
    #     self.control_input.speed = speed
    #     self.control_input.steer = steer
    #     self.control_input.brake = brake

    def Model_Predictive_Control(self, ego_ind):
        self.x.append(self.node.x)
        self.y.append(self.node.y)
        self.yaw.append(self.node.yaw)
        self.v.append(self.node.v)
        self.t.append(time)
        
        z_ref = calc_ref_trajectory_in_T_step(self.node, ego_ind, self.ref_path, self.sp) # 내 인덱스에서 미래 reference state
    
        steer_list = mpc_pure_pursuit(self.node, ego_ind, self.cx, self.cy, self.Parking_mission) # 내 위치에서 뽑아낸 미래 예측 steer들
        # print(steer_list)
        z_bar = mpc_predict_next_state(z_ref, self.node.x, self.node.y, self.node.yaw, self.node.v, self.v[-1], steer_list, 0) # 미래 예측된 state # 위에서 뽑은 걸 기반으로 뽑아낸 예측 state
        # print(steer_list)
        selected_index = mpc_cost_function(z_ref, z_bar, steer_list) # z_ref와 z_bar을 기반으로 해서 다음 state index 정하기
        # print(steer_list)
        self.node.update(10, steer_list[selected_index], 0)

        return z_ref[3, selected_index], steer_list[selected_index], 0, z_ref, selected_index
        
    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """
        print("self.cx:",self.cx[0],"node.x:",node.x)
        print("self.cy:",self.cy[0],"node.y:",node.y)
        dx = [node.x - x for x in self.cx[0 : -1]]
        dy = [node.y - y for y in self.cy[0 : -1]]
        # dx = [node.x - node.x]
        # dy = [node.y - node.y]
        # dx = [x for x in self.cx[0 : -1] - (x for x in self.cx[0 : -1])]
        # dy = [y for y in self.cy[0 : -1] - (y for y in self.cy[0 : -1])]
        dist = np.hypot(dx, dy)

        self.ind_old = int(np.argmin(dist))

        ind = self.ind_old

        return ind
    
    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y
        ################################################################### 선속도 계산
        dt = 0.2

        
        delta_x = self.current_postion.x - self.prev_x
        delta_y = self.current_postion.y - self.prev_y
        distance = math.sqrt(delta_x**2 + delta_y**2)
        self.current_vel = distance /dt  * 3.6

        self.prev_x = self.current_postion.x
        self.prev_y = self.current_postion.y
      

        # self.node.v = self.current_vel
            

       

        # print(self.current_postion.x,self.current_postion.y, self.vehicle_yaw)
    
    # def status_callback(self, msg):
    #     self.is_status = True
    #     self.current_vel = msg.velocity.x
    
    def load_global_path(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        global_path_x = []
        global_path_y = []
        global_path_yaw = []

        for key in sorted(data.keys(), key=int):
            point = data[key]
            global_path_x.append(point[0])
            global_path_y.append(point[1])
            global_path_yaw.append(point[2])

        return global_path_x, global_path_y, global_path_yaw
            
if __name__ == "__main__":
    morai = MPC().run()


import os
import numpy as np
import math
import random
import rospy
import copy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8

import ros_utils as sensors

from config import init_pos_set, map_tiles
import pickle


class AuvUwsim(object):
    
    def __init__(self):
        #Initialize sensors
        self.IG = sensors.imageGrabber()
        self.DI = sensors.depthImage()
        self.State_p = sensors.GetPose()
        self.State_v = sensors.GetVelocity()

        # Publisher
        self.Thruster_pub = rospy.Publisher("/vehicle1/thrusters_input",Float64MultiArray ,queue_size=1)
        self.reset_pub = rospy.Publisher("/vehicle1/resetdata",Odometry ,queue_size=1)
        self.pause_pub = rospy.Publisher("/pause",Int8,queue_size=1)

        # set initial variable
        self.init_pos_set = init_pos_set
        self.init_state = [0, 0, 7.4, 0, 0, 1.57, 0, 0.0,0.0,0.0,0.0,0.0]
        self.map_tiles = map_tiles   
    
    def state2msg(self, state):
        # convert state into a msg passed to set the vehicle
        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state
        #psi = -psi - np.pi/2
        
        msg = Odometry()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z # 4.5
        msg.pose.pose.orientation.w = np.cos(phi/2)*np.cos(psi/2)*np.cos(theta/2)+np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        msg.pose.pose.orientation.x = np.sin(phi/2)*np.cos(psi/2)*np.cos(theta/2)-np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) 
        msg.pose.pose.orientation.y = np.cos(phi/2)*np.cos(psi/2)*np.sin(theta/2)+np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
        msg.pose.pose.orientation.z = np.cos(phi/2)*np.sin(psi/2)*np.cos(theta/2)-np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

        msg.twist.twist.linear.x = u
        msg.twist.twist.linear.y = v
        msg.twist.twist.linear.z = w
        msg.twist.twist.angular.x = p
        msg.twist.twist.angular.y = q
        msg.twist.twist.angular.z = r
        return msg

    def action2msg(self, action):
        tau1, tau2 = action
        msg = Float64MultiArray()
        msg.data = [-tau1, -tau2, 0, 0, 0]
        return msg

    def get_retstate(self, state):
        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state
        return [x, y, psi, u, v, r]
   
    def generate_init_state(self):
        t_i = np.random.randint(0, len(self.map_tiles)) 
        t = self.map_tiles[-t_i]
        pos, heading = t.get_start_pos()
        return pos, heading


    def reset_sim(self):
        self.state = [0.0] * 12
        pos, heading = self.generate_init_state()
        self.state[:2], self.state[5] = pos, heading
        #set initial state
        msg = self.state2msg(self.state)
        self.reset_pub.publish(msg)

        #publish reset_flag
        flag = Int8()
        flag.data = 1
        self.pause_pub.publish(flag)

        ret_state = np.array(self.get_retstate(self.state))
        #ret_state = np.concatenate([ret_state, self.ang_enc], axis=-1)
        
        img = copy.deepcopy(self.IG.cv_image)
        
        return ret_state, self.IG.cv_image, {}

    def read_state(self):
        p = self.State_p.p
        v = self.State_v.v
        #p[-1] = -p[-1] - np.pi/2
        return np.append(p, v)

    def compute_reward(self, pos, heading, u):
        out_of_driving = True
        for t in self.map_tiles:
            if t.is_in(pos):
                out_of_driving = False
                dist, ang = t.get_pos_lane(pos, heading)
                reward = u * np.cos(ang) - 2.0 * abs(dist)
                break
        if out_of_driving:
            reward = -1000
        
        return reward, out_of_driving
        

    def frame_step(self, action):
        #publish action
        a_msg = self.action2msg(action)
        self.Thruster_pub.publish(a_msg)
        
        #run to next frame
        rospy.sleep(0.1)

        #subscribe new state
        self.state = self.read_state()
        # get reward
        img = copy.deepcopy(self.IG.cv_image)
        pos, heading, u = self.state[:2], self.state[5], self.state[6]
        
        reward, done = self.compute_reward(self.state[:2], heading, u)
                
        ret_state = self.get_retstate(self.state)
        
        return ret_state, img, reward, done, {}



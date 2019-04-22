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

from reward import cenline_extract, get_reward 
from env.config import pipeline_points


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

        # Initial point set
        #init_pos = []
        #for ps in pipeline_points:
        #    for p in ps:
        #        init_pos.append(p)
        #self.init_pos = init_pos
        self.init_pos = pipeline_points[0]
         

    
    def state2msg(self, state):
        # convert state into a msg passed to set the vehicle
        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state
        
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
        msg.data = [tau1, tau2, 0, 0, 0]
        return msg


    def get_retstate(self, state):
        x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state
        return [x, y, psi, u, v, r]
    
    def reset_sim(self, init_state = [-1.12,-3.6, 7.4,0,0, 1.27 ,0, 0.0,0.0,0.0,0.0,0.0]):
        self.state = init_state
        #self.state[5] = -np.arctan(1.8/0.56)
        self.state[5] = -0.3 - np.pi / 2
        rand_delta_ang = (random.random() * 2 - 1) * np.pi / 6
        self.state[5] += rand_delta_ang
        print("start at direction angle %f deg"%(self.state[5]/np.pi * 180))
        #self.state[:2] = random.choice(self.init_pos)
        
        
        #initialize flag variables
        self.step = 0
        self.last_rew = None
        self.last_feat = None
        self.failed_steps = 0
        
        #set initial state
        msg = self.state2msg(self.state)
        self.reset_pub.publish(msg)

        #publish reset_flag
        flag = Int8()
        flag.data = 1
        self.pause_pub.publish(flag)

        ret_state = np.array(self.get_retstate(self.state))
        
        img = copy.deepcopy(self.IG.cv_image)
        u = self.state[6]
        _, feat = get_reward(img, u)

        return ret_state, self.IG.cv_image, feat

    def frame_step(self, action):
        #publish action
        a_msg = self.action2msg(action)
        self.Thruster_pub.publish(a_msg)
        
        #run to next frame
        rospy.sleep(0.1)

        #subscribe new state
        self.state = np.append(self.State_p.p, self.State_v.v)

        # get reward
        img = copy.deepcopy(self.IG.cv_image)
        u = self.state[6]
        rew, feat = get_reward(img, u)

        done = False
        if rew is None or feat is None:
            rew = 0
            done = True


        # process None feat
        #if feat is None:
        #    if self.last_feat is not None:
        #        feat = self.last_feat
        #else:
        #    self.last_feat = feat
        
        # process the None reward
        #if rew is None:
        #    rew = 0
        #    self.failed_steps += 1
        #else:
        #    self.failed_steps = 0

        # judge end condition: fail_step > 5
        #done = False
        #if self.failed_steps > 5:   # if rew cannot be continuously detected by 5 times, end the episode.
        #    done = True 
        
        ret_state = self.get_retstate(self.state)
        
        self.step += 1
        return ret_state, img, rew, done, feat



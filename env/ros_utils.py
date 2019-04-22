import numpy as np
import math

import rospy
import roslaunch
import cv2
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8



class depthImage:
  def __init__(self):
    rospy.Subscriber("/vehicle1/rangecamera",Image,self.dimage_callback)
    self.bridge = CvBridge()
    self.height = -1
    self.width = -1
    self.depth_image = []
  def dimage_callback(self,data):
    try:
      self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
    except CvBridgeError as e:
      print(e)
    self.height, self.width = self.depth_image.shape
    #cv2.imshow("depth", self.depth_image/10) #normalization
    #cv2.waitKey(3)
  def getSize(self):
    return self.width,self.height

class imageGrabber:
  def __init__(self):
    rospy.Subscriber("vehicle1/camera1",Image,self.image_callback)
    self.bridge = CvBridge()
    self.height=-1
    self.width=-1
    self.channels=-1
    self.cv_image = []
  def image_callback(self,data):
    try:
      self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    self.height, self.width, self.channels = self.cv_image.shape
    #cv2.imshow("camera", self.cv_image)
    #cv2.waitKey(3)
  def getSize(self):
    return self.width,self.height

class GetPose:
  def __init__(self):
    rospy.Subscriber("/vehicle1/pose", Pose, self.p_callback)
    self.p = []
  def p_callback(self,data):
    q1 = data.orientation.x
    q2 = data.orientation.y
    q3 = data.orientation.z
    q4 = data.orientation.w
    newx = data.position.x
    newy = data.position.y
    newz = data.position.z
    newphi = math.atan2(2*(q1*q4+q2*q3),1-2*(q2*q2+q1*q1))
    newtheta = math.asin(2*(q2*q4-q1*q3))
    newpsi = math.atan2(2*(q3*q4+q1*q2),1-2*(q2*q2+q3*q3))
    self.p = [newx, newy, newz, newphi, newtheta, newpsi]

class GetVelocity:
  def __init__(self):
    rospy.Subscriber("/vehicle1/velocity", Twist, self.v_callback)
    self.v = []
  def v_callback(self,data):
    newu = data.linear.x
    newv = data.linear.y
    neww = data.linear.z
    newp = data.angular.x
    newq = data.angular.y
    newr = data.angular.z
    self.v = [newu, newv, neww, newp, newq, newr]

class Laser:
  def __init__(self):
    rospy.Subscriber("/vehicle1/multibeam", LaserScan, self.LS_callback)
    self.laser = []
  def LS_callback(self,data):
    self.laser = data.ranges 

class Trajectory:
#show given trajectory (update position)
  def __init__(self):
    self.tr_pub = rospy.Publisher("/vehicle2/dataNavigator", Odometry, queue_size=1)
  def tr_publish(self,data):
    x, y, z, phi, theta, psi = data
    t_msg = Odometry()
    t_msg.pose.pose.position.x = x
    t_msg.pose.pose.position.y = y
    t_msg.pose.pose.position.z = z
    t_msg.pose.pose.orientation.w = np.cos(phi/2)*np.cos(psi/2)*np.cos(theta/2)+np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    t_msg.pose.pose.orientation.x = np.sin(phi/2)*np.cos(psi/2)*np.cos(theta/2)-np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) 
    t_msg.pose.pose.orientation.y = np.cos(phi/2)*np.cos(psi/2)*np.sin(theta/2)+np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    t_msg.pose.pose.orientation.z = np.cos(phi/2)*np.sin(psi/2)*np.cos(theta/2)-np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    self.tr_pub.publish(t_msg)


def launch_from_py(node_name, filename):
    rospy.init_node(node_name, anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, [filename])
    return launch
 
    


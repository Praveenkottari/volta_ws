#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

aruco = cv2.aruco

camera_matrix = np.array([[980, 0, 360],
                          [0, 980, 480],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,), dtype=np.float32)

MARKER_LENGTH = 0.1

ARUCO_DICTS = {
    "DICT_5X5_50": aruco.DICT_5X5_50,
}

def estimate_pose_single_marker(corners, marker_length, camera_matrix, dist_coeffs):
    half_length = marker_length / 2.0
    object_points = np.array([
        [-half_length,  half_length, 0],
        [ half_length,  half_length, 0],
        [ half_length, -half_length, 0],
        [-half_length, -half_length, 0]
    ], dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not ret:
        rospy.logerr("solvePnP failed")
        return None, None
    return rvec, tvec

class ArucoPDController:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image, self.image_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTS["DICT_5X5_50"])
        try:
            self.parameters = aruco.DetectorParameters_create()
        except AttributeError:
            self.parameters = aruco.DetectorParameters()
        try:
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except AttributeError:
            rospy.logerr("cv2.aruco does not have ArucoDetector.")
            raise
        self.Kp_z = 0.2
        self.Kd_z = 0.1
        self.Kp_ang = 0.0005
        self.Kd_ang = 0.0001
        self.prev_error_z = 0.0
        self.prev_error_ang = 0.0
        self.prev_time = rospy.Time.now()
        rospy.loginfo("Aruco PD controller node started, subscribed to usb_cam/image_raw")

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))
            return
        (img_h, img_w) = frame.shape[:2]
        image_center_x = img_w / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        twist = Twist()
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            i = 0
            if hasattr(aruco, "estimatePoseSingleMarkers"):
                try:
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i], MARKER_LENGTH, camera_matrix, dist_coeffs
                    )
                    rvec = rvec[0][0]
                    tvec = tvec[0][0]
                except Exception as e:
                    rospy.logwarn("estimatePoseSingleMarkers error: {}. Falling back to solvePnP.".format(e))
                    corners_reshaped = corners[i].reshape((4, 2))
                    rvec, tvec = estimate_pose_single_marker(corners_reshaped, MARKER_LENGTH, camera_matrix, dist_coeffs)
                    if rvec is not None:
                        rvec = rvec.flatten()
                    if tvec is not None:
                        tvec = tvec.flatten()
            else:
                corners_reshaped = corners[i].reshape((4, 2))
                rvec, tvec = estimate_pose_single_marker(corners_reshaped, MARKER_LENGTH, camera_matrix, dist_coeffs)
                if rvec is not None:
                    rvec = rvec.flatten()
                if tvec is not None:
                    tvec = tvec.flatten()
            if rvec is None or tvec is None:
                rospy.logwarn("Pose estimation failed, not controlling this frame.")
            else:
                error_z = tvec[2] - 0.5
                current_time = rospy.Time.now()
                dt = (current_time - self.prev_time).to_sec()
                if dt == 0:
                    dt = 1e-6
                derivative_z = (error_z - self.prev_error_z) / dt
                linear_x = self.Kp_z * error_z + self.Kd_z * derivative_z
                marker_corners = corners[i].reshape((4, 2))
                marker_center_x = np.mean(marker_corners[:, 0])
                error_ang = marker_center_x - image_center_x
                derivative_ang = (error_ang - self.prev_error_ang) / dt
                angular_z = - (self.Kp_ang * error_ang + self.Kd_ang * derivative_ang)
                self.prev_error_z = error_z
                self.prev_error_ang = error_ang
                self.prev_time = current_time
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                rospy.loginfo("Control: linear_x: {:.2f}, angular_z: {:.2f}".format(linear_x, angular_z))
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        else:
            rospy.loginfo("No markers detected.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        cv2.imshow("Pose Estimation", frame)
        cv2.waitKey(1)

def main():
    rospy.init_node("aruco_pd_controller", anonymous=True)
    controller = ArucoPDController()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Aruco PD controller node")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
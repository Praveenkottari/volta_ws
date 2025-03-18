#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class YOLONode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)
        self.bridge = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5n.pt')
        rospy.loginfo("YOLOv5 model loaded successfully.")

        # Input topic for camera stream
        self.subscriber = rospy.Subscriber('/camera/camera', Image, self.callback)
        # Output topic for annotated images
        self.publisher = rospy.Publisher('/yolo/detections', Image, queue_size=10)

        # Video writer for saving the detections
        self.video_writer = None
        self.video_path = "./out1.avi"
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 20

    def callback(self, data):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Run YOLO inference
            results = self.model(cv_image)
            predictions = results.xyxy[0].cpu().numpy()

            # Annotate detections on the image
            for det in predictions:
                xmin, ymin, xmax, ymax, confidence, class_id = det[:6]
                label = self.model.names[int(class_id)]
                cv2.rectangle(cv_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{label} {confidence:.2f}", 
                            (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish annotated image
            annotated_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.publisher.publish(annotated_image)

            # Save the video
            if self.video_writer is None:
                self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'XVID'),
                                                    self.fps, (self.frame_width, self.frame_height))
            self.video_writer.write(cv_image)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def run(self):
        rospy.spin()
        if self.video_writer:
            self.video_writer.release()

if __name__ == "__main__":
    try:
        yolo_node = YOLONode()
        yolo_node.run()
    except rospy.ROSInterruptException:
        pass

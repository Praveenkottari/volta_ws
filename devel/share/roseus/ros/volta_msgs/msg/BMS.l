;; Auto-generated. Do not edit!


(when (boundp 'volta_msgs::BMS)
  (if (not (find-package "VOLTA_MSGS"))
    (make-package "VOLTA_MSGS"))
  (shadow 'BMS (find-package "VOLTA_MSGS")))
(unless (find-package "VOLTA_MSGS::BMS")
  (make-package "VOLTA_MSGS::BMS"))

(in-package "ROS")
;;//! \htmlinclude BMS.msg.html


(defclass volta_msgs::BMS
  :super ros::object
  :slots (_soc _soh _current _voltage ))

(defmethod volta_msgs::BMS
  (:init
   (&key
    ((:soc __soc) 0.0)
    ((:soh __soh) 0.0)
    ((:current __current) 0.0)
    ((:voltage __voltage) 0.0)
    )
   (send-super :init)
   (setq _soc (float __soc))
   (setq _soh (float __soh))
   (setq _current (float __current))
   (setq _voltage (float __voltage))
   self)
  (:soc
   (&optional __soc)
   (if __soc (setq _soc __soc)) _soc)
  (:soh
   (&optional __soh)
   (if __soh (setq _soh __soh)) _soh)
  (:current
   (&optional __current)
   (if __current (setq _current __current)) _current)
  (:voltage
   (&optional __voltage)
   (if __voltage (setq _voltage __voltage)) _voltage)
  (:serialization-length
   ()
   (+
    ;; float32 _soc
    4
    ;; float32 _soh
    4
    ;; float32 _current
    4
    ;; float32 _voltage
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _soc
       (sys::poke _soc (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _soh
       (sys::poke _soh (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _current
       (sys::poke _current (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _voltage
       (sys::poke _voltage (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _soc
     (setq _soc (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _soh
     (setq _soh (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _current
     (setq _current (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _voltage
     (setq _voltage (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get volta_msgs::BMS :md5sum-) "e5d0e1c804606ac58f230d85fba314ed")
(setf (get volta_msgs::BMS :datatype-) "volta_msgs/BMS")
(setf (get volta_msgs::BMS :definition-)
      "float32 soc
float32 soh
float32 current
float32 voltage

")



(provide :volta_msgs/BMS "e5d0e1c804606ac58f230d85fba314ed")



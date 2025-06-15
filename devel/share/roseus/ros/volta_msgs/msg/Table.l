;; Auto-generated. Do not edit!


(when (boundp 'volta_msgs::Table)
  (if (not (find-package "VOLTA_MSGS"))
    (make-package "VOLTA_MSGS"))
  (shadow 'Table (find-package "VOLTA_MSGS")))
(unless (find-package "VOLTA_MSGS::TABLE")
  (make-package "VOLTA_MSGS::TABLE"))

(in-package "ROS")
;;//! \htmlinclude Table.msg.html


(defclass volta_msgs::Table
  :super ros::object
  :slots (_soc _soh _current _voltage _system_status _relay_temp _powe_in_temp _motor_volt _battery_volt _hw_estop_volt _motorAlarm _diag_en _hw_Estop_state _sw_Estop_state _ros_Estop_state ))

(defmethod volta_msgs::Table
  (:init
   (&key
    ((:soc __soc) 0.0)
    ((:soh __soh) 0.0)
    ((:current __current) 0.0)
    ((:voltage __voltage) 0.0)
    ((:system_status __system_status) 0)
    ((:relay_temp __relay_temp) 0.0)
    ((:powe_in_temp __powe_in_temp) 0.0)
    ((:motor_volt __motor_volt) 0.0)
    ((:battery_volt __battery_volt) 0.0)
    ((:hw_estop_volt __hw_estop_volt) 0.0)
    ((:motorAlarm __motorAlarm) 0)
    ((:diag_en __diag_en) 0)
    ((:hw_Estop_state __hw_Estop_state) 0)
    ((:sw_Estop_state __sw_Estop_state) 0)
    ((:ros_Estop_state __ros_Estop_state) 0)
    )
   (send-super :init)
   (setq _soc (float __soc))
   (setq _soh (float __soh))
   (setq _current (float __current))
   (setq _voltage (float __voltage))
   (setq _system_status (round __system_status))
   (setq _relay_temp (float __relay_temp))
   (setq _powe_in_temp (float __powe_in_temp))
   (setq _motor_volt (float __motor_volt))
   (setq _battery_volt (float __battery_volt))
   (setq _hw_estop_volt (float __hw_estop_volt))
   (setq _motorAlarm (round __motorAlarm))
   (setq _diag_en (round __diag_en))
   (setq _hw_Estop_state (round __hw_Estop_state))
   (setq _sw_Estop_state (round __sw_Estop_state))
   (setq _ros_Estop_state (round __ros_Estop_state))
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
  (:system_status
   (&optional __system_status)
   (if __system_status (setq _system_status __system_status)) _system_status)
  (:relay_temp
   (&optional __relay_temp)
   (if __relay_temp (setq _relay_temp __relay_temp)) _relay_temp)
  (:powe_in_temp
   (&optional __powe_in_temp)
   (if __powe_in_temp (setq _powe_in_temp __powe_in_temp)) _powe_in_temp)
  (:motor_volt
   (&optional __motor_volt)
   (if __motor_volt (setq _motor_volt __motor_volt)) _motor_volt)
  (:battery_volt
   (&optional __battery_volt)
   (if __battery_volt (setq _battery_volt __battery_volt)) _battery_volt)
  (:hw_estop_volt
   (&optional __hw_estop_volt)
   (if __hw_estop_volt (setq _hw_estop_volt __hw_estop_volt)) _hw_estop_volt)
  (:motorAlarm
   (&optional __motorAlarm)
   (if __motorAlarm (setq _motorAlarm __motorAlarm)) _motorAlarm)
  (:diag_en
   (&optional __diag_en)
   (if __diag_en (setq _diag_en __diag_en)) _diag_en)
  (:hw_Estop_state
   (&optional __hw_Estop_state)
   (if __hw_Estop_state (setq _hw_Estop_state __hw_Estop_state)) _hw_Estop_state)
  (:sw_Estop_state
   (&optional __sw_Estop_state)
   (if __sw_Estop_state (setq _sw_Estop_state __sw_Estop_state)) _sw_Estop_state)
  (:ros_Estop_state
   (&optional __ros_Estop_state)
   (if __ros_Estop_state (setq _ros_Estop_state __ros_Estop_state)) _ros_Estop_state)
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
    ;; uint8 _system_status
    1
    ;; float32 _relay_temp
    4
    ;; float32 _powe_in_temp
    4
    ;; float32 _motor_volt
    4
    ;; float32 _battery_volt
    4
    ;; float32 _hw_estop_volt
    4
    ;; uint8 _motorAlarm
    1
    ;; uint8 _diag_en
    1
    ;; uint8 _hw_Estop_state
    1
    ;; uint8 _sw_Estop_state
    1
    ;; uint8 _ros_Estop_state
    1
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
     ;; uint8 _system_status
       (write-byte _system_status s)
     ;; float32 _relay_temp
       (sys::poke _relay_temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _powe_in_temp
       (sys::poke _powe_in_temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _motor_volt
       (sys::poke _motor_volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _battery_volt
       (sys::poke _battery_volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _hw_estop_volt
       (sys::poke _hw_estop_volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint8 _motorAlarm
       (write-byte _motorAlarm s)
     ;; uint8 _diag_en
       (write-byte _diag_en s)
     ;; uint8 _hw_Estop_state
       (write-byte _hw_Estop_state s)
     ;; uint8 _sw_Estop_state
       (write-byte _sw_Estop_state s)
     ;; uint8 _ros_Estop_state
       (write-byte _ros_Estop_state s)
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
   ;; uint8 _system_status
     (setq _system_status (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; float32 _relay_temp
     (setq _relay_temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _powe_in_temp
     (setq _powe_in_temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _motor_volt
     (setq _motor_volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _battery_volt
     (setq _battery_volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _hw_estop_volt
     (setq _hw_estop_volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint8 _motorAlarm
     (setq _motorAlarm (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _diag_en
     (setq _diag_en (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _hw_Estop_state
     (setq _hw_Estop_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _sw_Estop_state
     (setq _sw_Estop_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _ros_Estop_state
     (setq _ros_Estop_state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get volta_msgs::Table :md5sum-) "b849cd8a5229dad6cf68bbda639aeae9")
(setf (get volta_msgs::Table :datatype-) "volta_msgs/Table")
(setf (get volta_msgs::Table :definition-)
      "float32 soc
float32 soh
float32 current
float32 voltage
uint8 system_status
float32 relay_temp
float32 powe_in_temp

float32 motor_volt
float32 battery_volt
float32 hw_estop_volt
uint8 motorAlarm
uint8 diag_en
uint8 hw_Estop_state
uint8 sw_Estop_state
uint8 ros_Estop_state

")



(provide :volta_msgs/Table "b849cd8a5229dad6cf68bbda639aeae9")



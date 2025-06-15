;; Auto-generated. Do not edit!


(when (boundp 'volta_msgs::Diagnostics)
  (if (not (find-package "VOLTA_MSGS"))
    (make-package "VOLTA_MSGS"))
  (shadow 'Diagnostics (find-package "VOLTA_MSGS")))
(unless (find-package "VOLTA_MSGS::DIAGNOSTICS")
  (make-package "VOLTA_MSGS::DIAGNOSTICS"))

(in-package "ROS")
;;//! \htmlinclude Diagnostics.msg.html


(defclass volta_msgs::Diagnostics
  :super ros::object
  :slots (_LOWBATTERY_FLAG _CHARGING_FLAG _MOTOR_ALARM_FLAG _BMS_ALARM_FLAG _HIGH_TEMP_FLAG _COMM_ERROR_FLAG _BMS_SOC _BMS_SOH _BMS_Voltage _BMS_Current _BMS_voltageDiff _BMS_tempDiff _BMS_BCUMode _BMS_Alarm _M_State _M_Alarm _M_Power _M_Voltage _M_Current _M_Comm _Relay_Temp _PowerConn_Temp _PreCharge_Temp _Power_Supply_Temp _Battery_In_Volt _PRE_CHARGE_Volt _HW_ESTOP_Volt _SW_ESTOP_Volt _Motor_Alarm_Code _Motor_Fun_Fail _M_INIT _M_CW _M_CCW _M_STOP _M_BREAK _M_SETRPM _M_GETRPM _M_PARAM _M_ALARM _PRECHARGE_FUSE _MOTOR_FUSE _LED_STATE ))

(defmethod volta_msgs::Diagnostics
  (:init
   (&key
    ((:LOWBATTERY_FLAG __LOWBATTERY_FLAG) nil)
    ((:CHARGING_FLAG __CHARGING_FLAG) nil)
    ((:MOTOR_ALARM_FLAG __MOTOR_ALARM_FLAG) nil)
    ((:BMS_ALARM_FLAG __BMS_ALARM_FLAG) nil)
    ((:HIGH_TEMP_FLAG __HIGH_TEMP_FLAG) nil)
    ((:COMM_ERROR_FLAG __COMM_ERROR_FLAG) nil)
    ((:BMS_SOC __BMS_SOC) 0)
    ((:BMS_SOH __BMS_SOH) 0)
    ((:BMS_Voltage __BMS_Voltage) 0)
    ((:BMS_Current __BMS_Current) 0)
    ((:BMS_voltageDiff __BMS_voltageDiff) 0)
    ((:BMS_tempDiff __BMS_tempDiff) 0)
    ((:BMS_BCUMode __BMS_BCUMode) 0)
    ((:BMS_Alarm __BMS_Alarm) 0)
    ((:M_State __M_State) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_Alarm __M_Alarm) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_Power __M_Power) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_Voltage __M_Voltage) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_Current __M_Current) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_Comm __M_Comm) (make-array 2 :initial-element 0 :element-type :char))
    ((:Relay_Temp __Relay_Temp) 0.0)
    ((:PowerConn_Temp __PowerConn_Temp) 0.0)
    ((:PreCharge_Temp __PreCharge_Temp) 0.0)
    ((:Power_Supply_Temp __Power_Supply_Temp) 0.0)
    ((:Battery_In_Volt __Battery_In_Volt) 0.0)
    ((:PRE_CHARGE_Volt __PRE_CHARGE_Volt) 0.0)
    ((:HW_ESTOP_Volt __HW_ESTOP_Volt) 0.0)
    ((:SW_ESTOP_Volt __SW_ESTOP_Volt) 0.0)
    ((:Motor_Alarm_Code __Motor_Alarm_Code) (make-array 2 :initial-element 0 :element-type :integer))
    ((:Motor_Fun_Fail __Motor_Fun_Fail) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_INIT __M_INIT) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_CW __M_CW) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_CCW __M_CCW) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_STOP __M_STOP) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_BREAK __M_BREAK) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_SETRPM __M_SETRPM) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_GETRPM __M_GETRPM) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_PARAM __M_PARAM) (make-array 2 :initial-element 0 :element-type :integer))
    ((:M_ALARM __M_ALARM) (make-array 2 :initial-element 0 :element-type :integer))
    ((:PRECHARGE_FUSE __PRECHARGE_FUSE) 0)
    ((:MOTOR_FUSE __MOTOR_FUSE) 0)
    ((:LED_STATE __LED_STATE) 0)
    )
   (send-super :init)
   (setq _LOWBATTERY_FLAG __LOWBATTERY_FLAG)
   (setq _CHARGING_FLAG __CHARGING_FLAG)
   (setq _MOTOR_ALARM_FLAG __MOTOR_ALARM_FLAG)
   (setq _BMS_ALARM_FLAG __BMS_ALARM_FLAG)
   (setq _HIGH_TEMP_FLAG __HIGH_TEMP_FLAG)
   (setq _COMM_ERROR_FLAG __COMM_ERROR_FLAG)
   (setq _BMS_SOC (round __BMS_SOC))
   (setq _BMS_SOH (round __BMS_SOH))
   (setq _BMS_Voltage (round __BMS_Voltage))
   (setq _BMS_Current (round __BMS_Current))
   (setq _BMS_voltageDiff (round __BMS_voltageDiff))
   (setq _BMS_tempDiff (round __BMS_tempDiff))
   (setq _BMS_BCUMode (round __BMS_BCUMode))
   (setq _BMS_Alarm (round __BMS_Alarm))
   (setq _M_State __M_State)
   (setq _M_Alarm __M_Alarm)
   (setq _M_Power __M_Power)
   (setq _M_Voltage __M_Voltage)
   (setq _M_Current __M_Current)
   (setq _M_Comm __M_Comm)
   (setq _Relay_Temp (float __Relay_Temp))
   (setq _PowerConn_Temp (float __PowerConn_Temp))
   (setq _PreCharge_Temp (float __PreCharge_Temp))
   (setq _Power_Supply_Temp (float __Power_Supply_Temp))
   (setq _Battery_In_Volt (float __Battery_In_Volt))
   (setq _PRE_CHARGE_Volt (float __PRE_CHARGE_Volt))
   (setq _HW_ESTOP_Volt (float __HW_ESTOP_Volt))
   (setq _SW_ESTOP_Volt (float __SW_ESTOP_Volt))
   (setq _Motor_Alarm_Code __Motor_Alarm_Code)
   (setq _Motor_Fun_Fail __Motor_Fun_Fail)
   (setq _M_INIT __M_INIT)
   (setq _M_CW __M_CW)
   (setq _M_CCW __M_CCW)
   (setq _M_STOP __M_STOP)
   (setq _M_BREAK __M_BREAK)
   (setq _M_SETRPM __M_SETRPM)
   (setq _M_GETRPM __M_GETRPM)
   (setq _M_PARAM __M_PARAM)
   (setq _M_ALARM __M_ALARM)
   (setq _PRECHARGE_FUSE (round __PRECHARGE_FUSE))
   (setq _MOTOR_FUSE (round __MOTOR_FUSE))
   (setq _LED_STATE (round __LED_STATE))
   self)
  (:LOWBATTERY_FLAG
   (&optional (__LOWBATTERY_FLAG :null))
   (if (not (eq __LOWBATTERY_FLAG :null)) (setq _LOWBATTERY_FLAG __LOWBATTERY_FLAG)) _LOWBATTERY_FLAG)
  (:CHARGING_FLAG
   (&optional (__CHARGING_FLAG :null))
   (if (not (eq __CHARGING_FLAG :null)) (setq _CHARGING_FLAG __CHARGING_FLAG)) _CHARGING_FLAG)
  (:MOTOR_ALARM_FLAG
   (&optional (__MOTOR_ALARM_FLAG :null))
   (if (not (eq __MOTOR_ALARM_FLAG :null)) (setq _MOTOR_ALARM_FLAG __MOTOR_ALARM_FLAG)) _MOTOR_ALARM_FLAG)
  (:BMS_ALARM_FLAG
   (&optional (__BMS_ALARM_FLAG :null))
   (if (not (eq __BMS_ALARM_FLAG :null)) (setq _BMS_ALARM_FLAG __BMS_ALARM_FLAG)) _BMS_ALARM_FLAG)
  (:HIGH_TEMP_FLAG
   (&optional (__HIGH_TEMP_FLAG :null))
   (if (not (eq __HIGH_TEMP_FLAG :null)) (setq _HIGH_TEMP_FLAG __HIGH_TEMP_FLAG)) _HIGH_TEMP_FLAG)
  (:COMM_ERROR_FLAG
   (&optional (__COMM_ERROR_FLAG :null))
   (if (not (eq __COMM_ERROR_FLAG :null)) (setq _COMM_ERROR_FLAG __COMM_ERROR_FLAG)) _COMM_ERROR_FLAG)
  (:BMS_SOC
   (&optional __BMS_SOC)
   (if __BMS_SOC (setq _BMS_SOC __BMS_SOC)) _BMS_SOC)
  (:BMS_SOH
   (&optional __BMS_SOH)
   (if __BMS_SOH (setq _BMS_SOH __BMS_SOH)) _BMS_SOH)
  (:BMS_Voltage
   (&optional __BMS_Voltage)
   (if __BMS_Voltage (setq _BMS_Voltage __BMS_Voltage)) _BMS_Voltage)
  (:BMS_Current
   (&optional __BMS_Current)
   (if __BMS_Current (setq _BMS_Current __BMS_Current)) _BMS_Current)
  (:BMS_voltageDiff
   (&optional __BMS_voltageDiff)
   (if __BMS_voltageDiff (setq _BMS_voltageDiff __BMS_voltageDiff)) _BMS_voltageDiff)
  (:BMS_tempDiff
   (&optional __BMS_tempDiff)
   (if __BMS_tempDiff (setq _BMS_tempDiff __BMS_tempDiff)) _BMS_tempDiff)
  (:BMS_BCUMode
   (&optional __BMS_BCUMode)
   (if __BMS_BCUMode (setq _BMS_BCUMode __BMS_BCUMode)) _BMS_BCUMode)
  (:BMS_Alarm
   (&optional __BMS_Alarm)
   (if __BMS_Alarm (setq _BMS_Alarm __BMS_Alarm)) _BMS_Alarm)
  (:M_State
   (&optional __M_State)
   (if __M_State (setq _M_State __M_State)) _M_State)
  (:M_Alarm
   (&optional __M_Alarm)
   (if __M_Alarm (setq _M_Alarm __M_Alarm)) _M_Alarm)
  (:M_Power
   (&optional __M_Power)
   (if __M_Power (setq _M_Power __M_Power)) _M_Power)
  (:M_Voltage
   (&optional __M_Voltage)
   (if __M_Voltage (setq _M_Voltage __M_Voltage)) _M_Voltage)
  (:M_Current
   (&optional __M_Current)
   (if __M_Current (setq _M_Current __M_Current)) _M_Current)
  (:M_Comm
   (&optional __M_Comm)
   (if __M_Comm (setq _M_Comm __M_Comm)) _M_Comm)
  (:Relay_Temp
   (&optional __Relay_Temp)
   (if __Relay_Temp (setq _Relay_Temp __Relay_Temp)) _Relay_Temp)
  (:PowerConn_Temp
   (&optional __PowerConn_Temp)
   (if __PowerConn_Temp (setq _PowerConn_Temp __PowerConn_Temp)) _PowerConn_Temp)
  (:PreCharge_Temp
   (&optional __PreCharge_Temp)
   (if __PreCharge_Temp (setq _PreCharge_Temp __PreCharge_Temp)) _PreCharge_Temp)
  (:Power_Supply_Temp
   (&optional __Power_Supply_Temp)
   (if __Power_Supply_Temp (setq _Power_Supply_Temp __Power_Supply_Temp)) _Power_Supply_Temp)
  (:Battery_In_Volt
   (&optional __Battery_In_Volt)
   (if __Battery_In_Volt (setq _Battery_In_Volt __Battery_In_Volt)) _Battery_In_Volt)
  (:PRE_CHARGE_Volt
   (&optional __PRE_CHARGE_Volt)
   (if __PRE_CHARGE_Volt (setq _PRE_CHARGE_Volt __PRE_CHARGE_Volt)) _PRE_CHARGE_Volt)
  (:HW_ESTOP_Volt
   (&optional __HW_ESTOP_Volt)
   (if __HW_ESTOP_Volt (setq _HW_ESTOP_Volt __HW_ESTOP_Volt)) _HW_ESTOP_Volt)
  (:SW_ESTOP_Volt
   (&optional __SW_ESTOP_Volt)
   (if __SW_ESTOP_Volt (setq _SW_ESTOP_Volt __SW_ESTOP_Volt)) _SW_ESTOP_Volt)
  (:Motor_Alarm_Code
   (&optional __Motor_Alarm_Code)
   (if __Motor_Alarm_Code (setq _Motor_Alarm_Code __Motor_Alarm_Code)) _Motor_Alarm_Code)
  (:Motor_Fun_Fail
   (&optional __Motor_Fun_Fail)
   (if __Motor_Fun_Fail (setq _Motor_Fun_Fail __Motor_Fun_Fail)) _Motor_Fun_Fail)
  (:M_INIT
   (&optional __M_INIT)
   (if __M_INIT (setq _M_INIT __M_INIT)) _M_INIT)
  (:M_CW
   (&optional __M_CW)
   (if __M_CW (setq _M_CW __M_CW)) _M_CW)
  (:M_CCW
   (&optional __M_CCW)
   (if __M_CCW (setq _M_CCW __M_CCW)) _M_CCW)
  (:M_STOP
   (&optional __M_STOP)
   (if __M_STOP (setq _M_STOP __M_STOP)) _M_STOP)
  (:M_BREAK
   (&optional __M_BREAK)
   (if __M_BREAK (setq _M_BREAK __M_BREAK)) _M_BREAK)
  (:M_SETRPM
   (&optional __M_SETRPM)
   (if __M_SETRPM (setq _M_SETRPM __M_SETRPM)) _M_SETRPM)
  (:M_GETRPM
   (&optional __M_GETRPM)
   (if __M_GETRPM (setq _M_GETRPM __M_GETRPM)) _M_GETRPM)
  (:M_PARAM
   (&optional __M_PARAM)
   (if __M_PARAM (setq _M_PARAM __M_PARAM)) _M_PARAM)
  (:M_ALARM
   (&optional __M_ALARM)
   (if __M_ALARM (setq _M_ALARM __M_ALARM)) _M_ALARM)
  (:PRECHARGE_FUSE
   (&optional __PRECHARGE_FUSE)
   (if __PRECHARGE_FUSE (setq _PRECHARGE_FUSE __PRECHARGE_FUSE)) _PRECHARGE_FUSE)
  (:MOTOR_FUSE
   (&optional __MOTOR_FUSE)
   (if __MOTOR_FUSE (setq _MOTOR_FUSE __MOTOR_FUSE)) _MOTOR_FUSE)
  (:LED_STATE
   (&optional __LED_STATE)
   (if __LED_STATE (setq _LED_STATE __LED_STATE)) _LED_STATE)
  (:serialization-length
   ()
   (+
    ;; bool _LOWBATTERY_FLAG
    1
    ;; bool _CHARGING_FLAG
    1
    ;; bool _MOTOR_ALARM_FLAG
    1
    ;; bool _BMS_ALARM_FLAG
    1
    ;; bool _HIGH_TEMP_FLAG
    1
    ;; bool _COMM_ERROR_FLAG
    1
    ;; uint16 _BMS_SOC
    2
    ;; uint16 _BMS_SOH
    2
    ;; uint16 _BMS_Voltage
    2
    ;; int16 _BMS_Current
    2
    ;; uint16 _BMS_voltageDiff
    2
    ;; uint16 _BMS_tempDiff
    2
    ;; uint16 _BMS_BCUMode
    2
    ;; uint16 _BMS_Alarm
    2
    ;; uint16[2] _M_State
    (* 2    2)
    ;; uint16[2] _M_Alarm
    (* 2    2)
    ;; uint16[2] _M_Power
    (* 2    2)
    ;; uint16[2] _M_Voltage
    (* 2    2)
    ;; uint16[2] _M_Current
    (* 2    2)
    ;; uint8[2] _M_Comm
    (* 1    2)
    ;; float32 _Relay_Temp
    4
    ;; float32 _PowerConn_Temp
    4
    ;; float32 _PreCharge_Temp
    4
    ;; float32 _Power_Supply_Temp
    4
    ;; float32 _Battery_In_Volt
    4
    ;; float32 _PRE_CHARGE_Volt
    4
    ;; float32 _HW_ESTOP_Volt
    4
    ;; float32 _SW_ESTOP_Volt
    4
    ;; uint16[2] _Motor_Alarm_Code
    (* 2    2)
    ;; uint32[2] _Motor_Fun_Fail
    (* 4    2)
    ;; uint16[2] _M_INIT
    (* 2    2)
    ;; uint16[2] _M_CW
    (* 2    2)
    ;; uint16[2] _M_CCW
    (* 2    2)
    ;; uint16[2] _M_STOP
    (* 2    2)
    ;; uint16[2] _M_BREAK
    (* 2    2)
    ;; uint16[2] _M_SETRPM
    (* 2    2)
    ;; uint16[2] _M_GETRPM
    (* 2    2)
    ;; uint16[2] _M_PARAM
    (* 2    2)
    ;; uint16[2] _M_ALARM
    (* 2    2)
    ;; uint8 _PRECHARGE_FUSE
    1
    ;; uint8 _MOTOR_FUSE
    1
    ;; uint8 _LED_STATE
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _LOWBATTERY_FLAG
       (if _LOWBATTERY_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; bool _CHARGING_FLAG
       (if _CHARGING_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; bool _MOTOR_ALARM_FLAG
       (if _MOTOR_ALARM_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; bool _BMS_ALARM_FLAG
       (if _BMS_ALARM_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; bool _HIGH_TEMP_FLAG
       (if _HIGH_TEMP_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; bool _COMM_ERROR_FLAG
       (if _COMM_ERROR_FLAG (write-byte -1 s) (write-byte 0 s))
     ;; uint16 _BMS_SOC
       (write-word _BMS_SOC s)
     ;; uint16 _BMS_SOH
       (write-word _BMS_SOH s)
     ;; uint16 _BMS_Voltage
       (write-word _BMS_Voltage s)
     ;; int16 _BMS_Current
       (write-word _BMS_Current s)
     ;; uint16 _BMS_voltageDiff
       (write-word _BMS_voltageDiff s)
     ;; uint16 _BMS_tempDiff
       (write-word _BMS_tempDiff s)
     ;; uint16 _BMS_BCUMode
       (write-word _BMS_BCUMode s)
     ;; uint16 _BMS_Alarm
       (write-word _BMS_Alarm s)
     ;; uint16[2] _M_State
     (dotimes (i 2)
       (write-word (elt _M_State i) s)
       )
     ;; uint16[2] _M_Alarm
     (dotimes (i 2)
       (write-word (elt _M_Alarm i) s)
       )
     ;; uint16[2] _M_Power
     (dotimes (i 2)
       (write-word (elt _M_Power i) s)
       )
     ;; uint16[2] _M_Voltage
     (dotimes (i 2)
       (write-word (elt _M_Voltage i) s)
       )
     ;; uint16[2] _M_Current
     (dotimes (i 2)
       (write-word (elt _M_Current i) s)
       )
     ;; uint8[2] _M_Comm
     (princ _M_Comm s)
     ;; float32 _Relay_Temp
       (sys::poke _Relay_Temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _PowerConn_Temp
       (sys::poke _PowerConn_Temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _PreCharge_Temp
       (sys::poke _PreCharge_Temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Power_Supply_Temp
       (sys::poke _Power_Supply_Temp (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _Battery_In_Volt
       (sys::poke _Battery_In_Volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _PRE_CHARGE_Volt
       (sys::poke _PRE_CHARGE_Volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _HW_ESTOP_Volt
       (sys::poke _HW_ESTOP_Volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _SW_ESTOP_Volt
       (sys::poke _SW_ESTOP_Volt (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint16[2] _Motor_Alarm_Code
     (dotimes (i 2)
       (write-word (elt _Motor_Alarm_Code i) s)
       )
     ;; uint32[2] _Motor_Fun_Fail
     (dotimes (i 2)
       (write-long (elt _Motor_Fun_Fail i) s)
       )
     ;; uint16[2] _M_INIT
     (dotimes (i 2)
       (write-word (elt _M_INIT i) s)
       )
     ;; uint16[2] _M_CW
     (dotimes (i 2)
       (write-word (elt _M_CW i) s)
       )
     ;; uint16[2] _M_CCW
     (dotimes (i 2)
       (write-word (elt _M_CCW i) s)
       )
     ;; uint16[2] _M_STOP
     (dotimes (i 2)
       (write-word (elt _M_STOP i) s)
       )
     ;; uint16[2] _M_BREAK
     (dotimes (i 2)
       (write-word (elt _M_BREAK i) s)
       )
     ;; uint16[2] _M_SETRPM
     (dotimes (i 2)
       (write-word (elt _M_SETRPM i) s)
       )
     ;; uint16[2] _M_GETRPM
     (dotimes (i 2)
       (write-word (elt _M_GETRPM i) s)
       )
     ;; uint16[2] _M_PARAM
     (dotimes (i 2)
       (write-word (elt _M_PARAM i) s)
       )
     ;; uint16[2] _M_ALARM
     (dotimes (i 2)
       (write-word (elt _M_ALARM i) s)
       )
     ;; uint8 _PRECHARGE_FUSE
       (write-byte _PRECHARGE_FUSE s)
     ;; uint8 _MOTOR_FUSE
       (write-byte _MOTOR_FUSE s)
     ;; uint8 _LED_STATE
       (write-byte _LED_STATE s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _LOWBATTERY_FLAG
     (setq _LOWBATTERY_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _CHARGING_FLAG
     (setq _CHARGING_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _MOTOR_ALARM_FLAG
     (setq _MOTOR_ALARM_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _BMS_ALARM_FLAG
     (setq _BMS_ALARM_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _HIGH_TEMP_FLAG
     (setq _HIGH_TEMP_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; bool _COMM_ERROR_FLAG
     (setq _COMM_ERROR_FLAG (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; uint16 _BMS_SOC
     (setq _BMS_SOC (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_SOH
     (setq _BMS_SOH (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_Voltage
     (setq _BMS_Voltage (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _BMS_Current
     (setq _BMS_Current (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_voltageDiff
     (setq _BMS_voltageDiff (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_tempDiff
     (setq _BMS_tempDiff (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_BCUMode
     (setq _BMS_BCUMode (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _BMS_Alarm
     (setq _BMS_Alarm (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16[2] _M_State
   (dotimes (i (length _M_State))
     (setf (elt _M_State i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_Alarm
   (dotimes (i (length _M_Alarm))
     (setf (elt _M_Alarm i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_Power
   (dotimes (i (length _M_Power))
     (setf (elt _M_Power i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_Voltage
   (dotimes (i (length _M_Voltage))
     (setf (elt _M_Voltage i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_Current
   (dotimes (i (length _M_Current))
     (setf (elt _M_Current i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint8[2] _M_Comm
   (setq _M_Comm (make-array 2 :element-type :char))
   (replace _M_Comm buf :start2 ptr-) (incf ptr- 2)
   ;; float32 _Relay_Temp
     (setq _Relay_Temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _PowerConn_Temp
     (setq _PowerConn_Temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _PreCharge_Temp
     (setq _PreCharge_Temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Power_Supply_Temp
     (setq _Power_Supply_Temp (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _Battery_In_Volt
     (setq _Battery_In_Volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _PRE_CHARGE_Volt
     (setq _PRE_CHARGE_Volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _HW_ESTOP_Volt
     (setq _HW_ESTOP_Volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _SW_ESTOP_Volt
     (setq _SW_ESTOP_Volt (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint16[2] _Motor_Alarm_Code
   (dotimes (i (length _Motor_Alarm_Code))
     (setf (elt _Motor_Alarm_Code i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint32[2] _Motor_Fun_Fail
   (dotimes (i (length _Motor_Fun_Fail))
     (setf (elt _Motor_Fun_Fail i) (sys::peek buf ptr- :integer)) (incf ptr- 4)
     )
   ;; uint16[2] _M_INIT
   (dotimes (i (length _M_INIT))
     (setf (elt _M_INIT i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_CW
   (dotimes (i (length _M_CW))
     (setf (elt _M_CW i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_CCW
   (dotimes (i (length _M_CCW))
     (setf (elt _M_CCW i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_STOP
   (dotimes (i (length _M_STOP))
     (setf (elt _M_STOP i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_BREAK
   (dotimes (i (length _M_BREAK))
     (setf (elt _M_BREAK i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_SETRPM
   (dotimes (i (length _M_SETRPM))
     (setf (elt _M_SETRPM i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_GETRPM
   (dotimes (i (length _M_GETRPM))
     (setf (elt _M_GETRPM i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_PARAM
   (dotimes (i (length _M_PARAM))
     (setf (elt _M_PARAM i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint16[2] _M_ALARM
   (dotimes (i (length _M_ALARM))
     (setf (elt _M_ALARM i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     )
   ;; uint8 _PRECHARGE_FUSE
     (setq _PRECHARGE_FUSE (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _MOTOR_FUSE
     (setq _MOTOR_FUSE (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _LED_STATE
     (setq _LED_STATE (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(setf (get volta_msgs::Diagnostics :md5sum-) "724ceb6ccb5d72d3af9209169201c96c")
(setf (get volta_msgs::Diagnostics :datatype-) "volta_msgs/Diagnostics")
(setf (get volta_msgs::Diagnostics :definition-)
      "bool LOWBATTERY_FLAG
bool CHARGING_FLAG
bool MOTOR_ALARM_FLAG
bool BMS_ALARM_FLAG
bool HIGH_TEMP_FLAG
bool COMM_ERROR_FLAG

uint16 BMS_SOC
uint16 BMS_SOH
uint16 BMS_Voltage
int16 BMS_Current
uint16 BMS_voltageDiff
uint16 BMS_tempDiff
uint16 BMS_BCUMode
uint16 BMS_Alarm

uint16[2] M_State
uint16[2] M_Alarm
uint16[2] M_Power
uint16[2] M_Voltage
uint16[2] M_Current
uint8[2] M_Comm

float32 Relay_Temp
float32 PowerConn_Temp
float32 PreCharge_Temp
float32 Power_Supply_Temp


float32 Battery_In_Volt
float32 PRE_CHARGE_Volt
float32 HW_ESTOP_Volt
float32 SW_ESTOP_Volt

uint16[2] Motor_Alarm_Code
uint32[2] Motor_Fun_Fail

uint16[2] M_INIT
uint16[2] M_CW
uint16[2] M_CCW
uint16[2] M_STOP
uint16[2] M_BREAK
uint16[2] M_SETRPM
uint16[2] M_GETRPM
uint16[2] M_PARAM
uint16[2] M_ALARM

uint8 PRECHARGE_FUSE
uint8 MOTOR_FUSE
uint8 LED_STATE

")



(provide :volta_msgs/Diagnostics "724ceb6ccb5d72d3af9209169201c96c")



; Auto-generated. Do not edit!


(cl:in-package volta_msgs-msg)


;//! \htmlinclude Diagnostics.msg.html

(cl:defclass <Diagnostics> (roslisp-msg-protocol:ros-message)
  ((LOWBATTERY_FLAG
    :reader LOWBATTERY_FLAG
    :initarg :LOWBATTERY_FLAG
    :type cl:boolean
    :initform cl:nil)
   (CHARGING_FLAG
    :reader CHARGING_FLAG
    :initarg :CHARGING_FLAG
    :type cl:boolean
    :initform cl:nil)
   (MOTOR_ALARM_FLAG
    :reader MOTOR_ALARM_FLAG
    :initarg :MOTOR_ALARM_FLAG
    :type cl:boolean
    :initform cl:nil)
   (BMS_ALARM_FLAG
    :reader BMS_ALARM_FLAG
    :initarg :BMS_ALARM_FLAG
    :type cl:boolean
    :initform cl:nil)
   (HIGH_TEMP_FLAG
    :reader HIGH_TEMP_FLAG
    :initarg :HIGH_TEMP_FLAG
    :type cl:boolean
    :initform cl:nil)
   (COMM_ERROR_FLAG
    :reader COMM_ERROR_FLAG
    :initarg :COMM_ERROR_FLAG
    :type cl:boolean
    :initform cl:nil)
   (BMS_SOC
    :reader BMS_SOC
    :initarg :BMS_SOC
    :type cl:fixnum
    :initform 0)
   (BMS_SOH
    :reader BMS_SOH
    :initarg :BMS_SOH
    :type cl:fixnum
    :initform 0)
   (BMS_Voltage
    :reader BMS_Voltage
    :initarg :BMS_Voltage
    :type cl:fixnum
    :initform 0)
   (BMS_Current
    :reader BMS_Current
    :initarg :BMS_Current
    :type cl:fixnum
    :initform 0)
   (BMS_voltageDiff
    :reader BMS_voltageDiff
    :initarg :BMS_voltageDiff
    :type cl:fixnum
    :initform 0)
   (BMS_tempDiff
    :reader BMS_tempDiff
    :initarg :BMS_tempDiff
    :type cl:fixnum
    :initform 0)
   (BMS_BCUMode
    :reader BMS_BCUMode
    :initarg :BMS_BCUMode
    :type cl:fixnum
    :initform 0)
   (BMS_Alarm
    :reader BMS_Alarm
    :initarg :BMS_Alarm
    :type cl:fixnum
    :initform 0)
   (M_State
    :reader M_State
    :initarg :M_State
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_Alarm
    :reader M_Alarm
    :initarg :M_Alarm
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_Power
    :reader M_Power
    :initarg :M_Power
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_Voltage
    :reader M_Voltage
    :initarg :M_Voltage
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_Current
    :reader M_Current
    :initarg :M_Current
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_Comm
    :reader M_Comm
    :initarg :M_Comm
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (Relay_Temp
    :reader Relay_Temp
    :initarg :Relay_Temp
    :type cl:float
    :initform 0.0)
   (PowerConn_Temp
    :reader PowerConn_Temp
    :initarg :PowerConn_Temp
    :type cl:float
    :initform 0.0)
   (PreCharge_Temp
    :reader PreCharge_Temp
    :initarg :PreCharge_Temp
    :type cl:float
    :initform 0.0)
   (Power_Supply_Temp
    :reader Power_Supply_Temp
    :initarg :Power_Supply_Temp
    :type cl:float
    :initform 0.0)
   (Battery_In_Volt
    :reader Battery_In_Volt
    :initarg :Battery_In_Volt
    :type cl:float
    :initform 0.0)
   (PRE_CHARGE_Volt
    :reader PRE_CHARGE_Volt
    :initarg :PRE_CHARGE_Volt
    :type cl:float
    :initform 0.0)
   (HW_ESTOP_Volt
    :reader HW_ESTOP_Volt
    :initarg :HW_ESTOP_Volt
    :type cl:float
    :initform 0.0)
   (SW_ESTOP_Volt
    :reader SW_ESTOP_Volt
    :initarg :SW_ESTOP_Volt
    :type cl:float
    :initform 0.0)
   (Motor_Alarm_Code
    :reader Motor_Alarm_Code
    :initarg :Motor_Alarm_Code
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (Motor_Fun_Fail
    :reader Motor_Fun_Fail
    :initarg :Motor_Fun_Fail
    :type (cl:vector cl:integer)
   :initform (cl:make-array 2 :element-type 'cl:integer :initial-element 0))
   (M_INIT
    :reader M_INIT
    :initarg :M_INIT
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_CW
    :reader M_CW
    :initarg :M_CW
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_CCW
    :reader M_CCW
    :initarg :M_CCW
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_STOP
    :reader M_STOP
    :initarg :M_STOP
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_BREAK
    :reader M_BREAK
    :initarg :M_BREAK
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_SETRPM
    :reader M_SETRPM
    :initarg :M_SETRPM
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_GETRPM
    :reader M_GETRPM
    :initarg :M_GETRPM
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_PARAM
    :reader M_PARAM
    :initarg :M_PARAM
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (M_ALARM
    :reader M_ALARM
    :initarg :M_ALARM
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 2 :element-type 'cl:fixnum :initial-element 0))
   (PRECHARGE_FUSE
    :reader PRECHARGE_FUSE
    :initarg :PRECHARGE_FUSE
    :type cl:fixnum
    :initform 0)
   (MOTOR_FUSE
    :reader MOTOR_FUSE
    :initarg :MOTOR_FUSE
    :type cl:fixnum
    :initform 0)
   (LED_STATE
    :reader LED_STATE
    :initarg :LED_STATE
    :type cl:fixnum
    :initform 0))
)

(cl:defclass Diagnostics (<Diagnostics>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Diagnostics>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Diagnostics)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name volta_msgs-msg:<Diagnostics> is deprecated: use volta_msgs-msg:Diagnostics instead.")))

(cl:ensure-generic-function 'LOWBATTERY_FLAG-val :lambda-list '(m))
(cl:defmethod LOWBATTERY_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:LOWBATTERY_FLAG-val is deprecated.  Use volta_msgs-msg:LOWBATTERY_FLAG instead.")
  (LOWBATTERY_FLAG m))

(cl:ensure-generic-function 'CHARGING_FLAG-val :lambda-list '(m))
(cl:defmethod CHARGING_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:CHARGING_FLAG-val is deprecated.  Use volta_msgs-msg:CHARGING_FLAG instead.")
  (CHARGING_FLAG m))

(cl:ensure-generic-function 'MOTOR_ALARM_FLAG-val :lambda-list '(m))
(cl:defmethod MOTOR_ALARM_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:MOTOR_ALARM_FLAG-val is deprecated.  Use volta_msgs-msg:MOTOR_ALARM_FLAG instead.")
  (MOTOR_ALARM_FLAG m))

(cl:ensure-generic-function 'BMS_ALARM_FLAG-val :lambda-list '(m))
(cl:defmethod BMS_ALARM_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_ALARM_FLAG-val is deprecated.  Use volta_msgs-msg:BMS_ALARM_FLAG instead.")
  (BMS_ALARM_FLAG m))

(cl:ensure-generic-function 'HIGH_TEMP_FLAG-val :lambda-list '(m))
(cl:defmethod HIGH_TEMP_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:HIGH_TEMP_FLAG-val is deprecated.  Use volta_msgs-msg:HIGH_TEMP_FLAG instead.")
  (HIGH_TEMP_FLAG m))

(cl:ensure-generic-function 'COMM_ERROR_FLAG-val :lambda-list '(m))
(cl:defmethod COMM_ERROR_FLAG-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:COMM_ERROR_FLAG-val is deprecated.  Use volta_msgs-msg:COMM_ERROR_FLAG instead.")
  (COMM_ERROR_FLAG m))

(cl:ensure-generic-function 'BMS_SOC-val :lambda-list '(m))
(cl:defmethod BMS_SOC-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_SOC-val is deprecated.  Use volta_msgs-msg:BMS_SOC instead.")
  (BMS_SOC m))

(cl:ensure-generic-function 'BMS_SOH-val :lambda-list '(m))
(cl:defmethod BMS_SOH-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_SOH-val is deprecated.  Use volta_msgs-msg:BMS_SOH instead.")
  (BMS_SOH m))

(cl:ensure-generic-function 'BMS_Voltage-val :lambda-list '(m))
(cl:defmethod BMS_Voltage-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_Voltage-val is deprecated.  Use volta_msgs-msg:BMS_Voltage instead.")
  (BMS_Voltage m))

(cl:ensure-generic-function 'BMS_Current-val :lambda-list '(m))
(cl:defmethod BMS_Current-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_Current-val is deprecated.  Use volta_msgs-msg:BMS_Current instead.")
  (BMS_Current m))

(cl:ensure-generic-function 'BMS_voltageDiff-val :lambda-list '(m))
(cl:defmethod BMS_voltageDiff-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_voltageDiff-val is deprecated.  Use volta_msgs-msg:BMS_voltageDiff instead.")
  (BMS_voltageDiff m))

(cl:ensure-generic-function 'BMS_tempDiff-val :lambda-list '(m))
(cl:defmethod BMS_tempDiff-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_tempDiff-val is deprecated.  Use volta_msgs-msg:BMS_tempDiff instead.")
  (BMS_tempDiff m))

(cl:ensure-generic-function 'BMS_BCUMode-val :lambda-list '(m))
(cl:defmethod BMS_BCUMode-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_BCUMode-val is deprecated.  Use volta_msgs-msg:BMS_BCUMode instead.")
  (BMS_BCUMode m))

(cl:ensure-generic-function 'BMS_Alarm-val :lambda-list '(m))
(cl:defmethod BMS_Alarm-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:BMS_Alarm-val is deprecated.  Use volta_msgs-msg:BMS_Alarm instead.")
  (BMS_Alarm m))

(cl:ensure-generic-function 'M_State-val :lambda-list '(m))
(cl:defmethod M_State-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_State-val is deprecated.  Use volta_msgs-msg:M_State instead.")
  (M_State m))

(cl:ensure-generic-function 'M_Alarm-val :lambda-list '(m))
(cl:defmethod M_Alarm-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_Alarm-val is deprecated.  Use volta_msgs-msg:M_Alarm instead.")
  (M_Alarm m))

(cl:ensure-generic-function 'M_Power-val :lambda-list '(m))
(cl:defmethod M_Power-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_Power-val is deprecated.  Use volta_msgs-msg:M_Power instead.")
  (M_Power m))

(cl:ensure-generic-function 'M_Voltage-val :lambda-list '(m))
(cl:defmethod M_Voltage-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_Voltage-val is deprecated.  Use volta_msgs-msg:M_Voltage instead.")
  (M_Voltage m))

(cl:ensure-generic-function 'M_Current-val :lambda-list '(m))
(cl:defmethod M_Current-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_Current-val is deprecated.  Use volta_msgs-msg:M_Current instead.")
  (M_Current m))

(cl:ensure-generic-function 'M_Comm-val :lambda-list '(m))
(cl:defmethod M_Comm-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_Comm-val is deprecated.  Use volta_msgs-msg:M_Comm instead.")
  (M_Comm m))

(cl:ensure-generic-function 'Relay_Temp-val :lambda-list '(m))
(cl:defmethod Relay_Temp-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:Relay_Temp-val is deprecated.  Use volta_msgs-msg:Relay_Temp instead.")
  (Relay_Temp m))

(cl:ensure-generic-function 'PowerConn_Temp-val :lambda-list '(m))
(cl:defmethod PowerConn_Temp-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:PowerConn_Temp-val is deprecated.  Use volta_msgs-msg:PowerConn_Temp instead.")
  (PowerConn_Temp m))

(cl:ensure-generic-function 'PreCharge_Temp-val :lambda-list '(m))
(cl:defmethod PreCharge_Temp-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:PreCharge_Temp-val is deprecated.  Use volta_msgs-msg:PreCharge_Temp instead.")
  (PreCharge_Temp m))

(cl:ensure-generic-function 'Power_Supply_Temp-val :lambda-list '(m))
(cl:defmethod Power_Supply_Temp-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:Power_Supply_Temp-val is deprecated.  Use volta_msgs-msg:Power_Supply_Temp instead.")
  (Power_Supply_Temp m))

(cl:ensure-generic-function 'Battery_In_Volt-val :lambda-list '(m))
(cl:defmethod Battery_In_Volt-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:Battery_In_Volt-val is deprecated.  Use volta_msgs-msg:Battery_In_Volt instead.")
  (Battery_In_Volt m))

(cl:ensure-generic-function 'PRE_CHARGE_Volt-val :lambda-list '(m))
(cl:defmethod PRE_CHARGE_Volt-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:PRE_CHARGE_Volt-val is deprecated.  Use volta_msgs-msg:PRE_CHARGE_Volt instead.")
  (PRE_CHARGE_Volt m))

(cl:ensure-generic-function 'HW_ESTOP_Volt-val :lambda-list '(m))
(cl:defmethod HW_ESTOP_Volt-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:HW_ESTOP_Volt-val is deprecated.  Use volta_msgs-msg:HW_ESTOP_Volt instead.")
  (HW_ESTOP_Volt m))

(cl:ensure-generic-function 'SW_ESTOP_Volt-val :lambda-list '(m))
(cl:defmethod SW_ESTOP_Volt-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:SW_ESTOP_Volt-val is deprecated.  Use volta_msgs-msg:SW_ESTOP_Volt instead.")
  (SW_ESTOP_Volt m))

(cl:ensure-generic-function 'Motor_Alarm_Code-val :lambda-list '(m))
(cl:defmethod Motor_Alarm_Code-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:Motor_Alarm_Code-val is deprecated.  Use volta_msgs-msg:Motor_Alarm_Code instead.")
  (Motor_Alarm_Code m))

(cl:ensure-generic-function 'Motor_Fun_Fail-val :lambda-list '(m))
(cl:defmethod Motor_Fun_Fail-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:Motor_Fun_Fail-val is deprecated.  Use volta_msgs-msg:Motor_Fun_Fail instead.")
  (Motor_Fun_Fail m))

(cl:ensure-generic-function 'M_INIT-val :lambda-list '(m))
(cl:defmethod M_INIT-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_INIT-val is deprecated.  Use volta_msgs-msg:M_INIT instead.")
  (M_INIT m))

(cl:ensure-generic-function 'M_CW-val :lambda-list '(m))
(cl:defmethod M_CW-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_CW-val is deprecated.  Use volta_msgs-msg:M_CW instead.")
  (M_CW m))

(cl:ensure-generic-function 'M_CCW-val :lambda-list '(m))
(cl:defmethod M_CCW-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_CCW-val is deprecated.  Use volta_msgs-msg:M_CCW instead.")
  (M_CCW m))

(cl:ensure-generic-function 'M_STOP-val :lambda-list '(m))
(cl:defmethod M_STOP-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_STOP-val is deprecated.  Use volta_msgs-msg:M_STOP instead.")
  (M_STOP m))

(cl:ensure-generic-function 'M_BREAK-val :lambda-list '(m))
(cl:defmethod M_BREAK-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_BREAK-val is deprecated.  Use volta_msgs-msg:M_BREAK instead.")
  (M_BREAK m))

(cl:ensure-generic-function 'M_SETRPM-val :lambda-list '(m))
(cl:defmethod M_SETRPM-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_SETRPM-val is deprecated.  Use volta_msgs-msg:M_SETRPM instead.")
  (M_SETRPM m))

(cl:ensure-generic-function 'M_GETRPM-val :lambda-list '(m))
(cl:defmethod M_GETRPM-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_GETRPM-val is deprecated.  Use volta_msgs-msg:M_GETRPM instead.")
  (M_GETRPM m))

(cl:ensure-generic-function 'M_PARAM-val :lambda-list '(m))
(cl:defmethod M_PARAM-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_PARAM-val is deprecated.  Use volta_msgs-msg:M_PARAM instead.")
  (M_PARAM m))

(cl:ensure-generic-function 'M_ALARM-val :lambda-list '(m))
(cl:defmethod M_ALARM-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:M_ALARM-val is deprecated.  Use volta_msgs-msg:M_ALARM instead.")
  (M_ALARM m))

(cl:ensure-generic-function 'PRECHARGE_FUSE-val :lambda-list '(m))
(cl:defmethod PRECHARGE_FUSE-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:PRECHARGE_FUSE-val is deprecated.  Use volta_msgs-msg:PRECHARGE_FUSE instead.")
  (PRECHARGE_FUSE m))

(cl:ensure-generic-function 'MOTOR_FUSE-val :lambda-list '(m))
(cl:defmethod MOTOR_FUSE-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:MOTOR_FUSE-val is deprecated.  Use volta_msgs-msg:MOTOR_FUSE instead.")
  (MOTOR_FUSE m))

(cl:ensure-generic-function 'LED_STATE-val :lambda-list '(m))
(cl:defmethod LED_STATE-val ((m <Diagnostics>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:LED_STATE-val is deprecated.  Use volta_msgs-msg:LED_STATE instead.")
  (LED_STATE m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Diagnostics>) ostream)
  "Serializes a message object of type '<Diagnostics>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'LOWBATTERY_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'CHARGING_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'MOTOR_ALARM_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'BMS_ALARM_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'HIGH_TEMP_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'COMM_ERROR_FLAG) 1 0)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_SOC)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_SOC)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_SOH)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_SOH)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_Voltage)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_Voltage)) ostream)
  (cl:let* ((signed (cl:slot-value msg 'BMS_Current)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_voltageDiff)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_voltageDiff)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_tempDiff)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_tempDiff)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_BCUMode)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_BCUMode)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_Alarm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_Alarm)) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_State))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_Alarm))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_Power))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_Voltage))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_Current))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'M_Comm))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Relay_Temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'PowerConn_Temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'PreCharge_Temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Power_Supply_Temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'Battery_In_Volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'PRE_CHARGE_Volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'HW_ESTOP_Volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'SW_ESTOP_Volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'Motor_Alarm_Code))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) ele) ostream))
   (cl:slot-value msg 'Motor_Fun_Fail))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_INIT))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_CW))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_CCW))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_STOP))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_BREAK))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_SETRPM))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_GETRPM))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_PARAM))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream))
   (cl:slot-value msg 'M_ALARM))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'PRECHARGE_FUSE)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'MOTOR_FUSE)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'LED_STATE)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Diagnostics>) istream)
  "Deserializes a message object of type '<Diagnostics>"
    (cl:setf (cl:slot-value msg 'LOWBATTERY_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'CHARGING_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'MOTOR_ALARM_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'BMS_ALARM_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'HIGH_TEMP_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:slot-value msg 'COMM_ERROR_FLAG) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_SOC)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_SOC)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_SOH)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_SOH)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_Voltage)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_Voltage)) (cl:read-byte istream))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'BMS_Current) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_voltageDiff)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_voltageDiff)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_tempDiff)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_tempDiff)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_BCUMode)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_BCUMode)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'BMS_Alarm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'BMS_Alarm)) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'M_State) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_State)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_Alarm) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_Alarm)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_Power) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_Power)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_Voltage) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_Voltage)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_Current) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_Current)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_Comm) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_Comm)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Relay_Temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'PowerConn_Temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'PreCharge_Temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Power_Supply_Temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'Battery_In_Volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'PRE_CHARGE_Volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'HW_ESTOP_Volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'SW_ESTOP_Volt) (roslisp-utils:decode-single-float-bits bits)))
  (cl:setf (cl:slot-value msg 'Motor_Alarm_Code) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'Motor_Alarm_Code)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'Motor_Fun_Fail) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'Motor_Fun_Fail)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_INIT) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_INIT)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_CW) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_CW)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_CCW) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_CCW)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_STOP) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_STOP)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_BREAK) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_BREAK)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_SETRPM) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_SETRPM)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_GETRPM) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_GETRPM)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_PARAM) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_PARAM)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
  (cl:setf (cl:slot-value msg 'M_ALARM) (cl:make-array 2))
  (cl:let ((vals (cl:slot-value msg 'M_ALARM)))
    (cl:dotimes (i 2)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'PRECHARGE_FUSE)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'MOTOR_FUSE)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'LED_STATE)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Diagnostics>)))
  "Returns string type for a message object of type '<Diagnostics>"
  "volta_msgs/Diagnostics")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Diagnostics)))
  "Returns string type for a message object of type 'Diagnostics"
  "volta_msgs/Diagnostics")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Diagnostics>)))
  "Returns md5sum for a message object of type '<Diagnostics>"
  "724ceb6ccb5d72d3af9209169201c96c")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Diagnostics)))
  "Returns md5sum for a message object of type 'Diagnostics"
  "724ceb6ccb5d72d3af9209169201c96c")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Diagnostics>)))
  "Returns full string definition for message of type '<Diagnostics>"
  (cl:format cl:nil "bool LOWBATTERY_FLAG~%bool CHARGING_FLAG~%bool MOTOR_ALARM_FLAG~%bool BMS_ALARM_FLAG~%bool HIGH_TEMP_FLAG~%bool COMM_ERROR_FLAG~%~%uint16 BMS_SOC~%uint16 BMS_SOH~%uint16 BMS_Voltage~%int16 BMS_Current~%uint16 BMS_voltageDiff~%uint16 BMS_tempDiff~%uint16 BMS_BCUMode~%uint16 BMS_Alarm~%~%uint16[2] M_State~%uint16[2] M_Alarm~%uint16[2] M_Power~%uint16[2] M_Voltage~%uint16[2] M_Current~%uint8[2] M_Comm~%~%float32 Relay_Temp~%float32 PowerConn_Temp~%float32 PreCharge_Temp~%float32 Power_Supply_Temp~%~%~%float32 Battery_In_Volt~%float32 PRE_CHARGE_Volt~%float32 HW_ESTOP_Volt~%float32 SW_ESTOP_Volt~%~%uint16[2] Motor_Alarm_Code~%uint32[2] Motor_Fun_Fail~%~%uint16[2] M_INIT~%uint16[2] M_CW~%uint16[2] M_CCW~%uint16[2] M_STOP~%uint16[2] M_BREAK~%uint16[2] M_SETRPM~%uint16[2] M_GETRPM~%uint16[2] M_PARAM~%uint16[2] M_ALARM~%~%uint8 PRECHARGE_FUSE~%uint8 MOTOR_FUSE~%uint8 LED_STATE~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Diagnostics)))
  "Returns full string definition for message of type 'Diagnostics"
  (cl:format cl:nil "bool LOWBATTERY_FLAG~%bool CHARGING_FLAG~%bool MOTOR_ALARM_FLAG~%bool BMS_ALARM_FLAG~%bool HIGH_TEMP_FLAG~%bool COMM_ERROR_FLAG~%~%uint16 BMS_SOC~%uint16 BMS_SOH~%uint16 BMS_Voltage~%int16 BMS_Current~%uint16 BMS_voltageDiff~%uint16 BMS_tempDiff~%uint16 BMS_BCUMode~%uint16 BMS_Alarm~%~%uint16[2] M_State~%uint16[2] M_Alarm~%uint16[2] M_Power~%uint16[2] M_Voltage~%uint16[2] M_Current~%uint8[2] M_Comm~%~%float32 Relay_Temp~%float32 PowerConn_Temp~%float32 PreCharge_Temp~%float32 Power_Supply_Temp~%~%~%float32 Battery_In_Volt~%float32 PRE_CHARGE_Volt~%float32 HW_ESTOP_Volt~%float32 SW_ESTOP_Volt~%~%uint16[2] Motor_Alarm_Code~%uint32[2] Motor_Fun_Fail~%~%uint16[2] M_INIT~%uint16[2] M_CW~%uint16[2] M_CCW~%uint16[2] M_STOP~%uint16[2] M_BREAK~%uint16[2] M_SETRPM~%uint16[2] M_GETRPM~%uint16[2] M_PARAM~%uint16[2] M_ALARM~%~%uint8 PRECHARGE_FUSE~%uint8 MOTOR_FUSE~%uint8 LED_STATE~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Diagnostics>))
  (cl:+ 0
     1
     1
     1
     1
     1
     1
     2
     2
     2
     2
     2
     2
     2
     2
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_State) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_Alarm) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_Power) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_Voltage) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_Current) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_Comm) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
     4
     4
     4
     4
     4
     4
     4
     4
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Motor_Alarm_Code) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'Motor_Fun_Fail) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_INIT) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_CW) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_CCW) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_STOP) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_BREAK) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_SETRPM) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_GETRPM) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_PARAM) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'M_ALARM) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 2)))
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Diagnostics>))
  "Converts a ROS message object to a list"
  (cl:list 'Diagnostics
    (cl:cons ':LOWBATTERY_FLAG (LOWBATTERY_FLAG msg))
    (cl:cons ':CHARGING_FLAG (CHARGING_FLAG msg))
    (cl:cons ':MOTOR_ALARM_FLAG (MOTOR_ALARM_FLAG msg))
    (cl:cons ':BMS_ALARM_FLAG (BMS_ALARM_FLAG msg))
    (cl:cons ':HIGH_TEMP_FLAG (HIGH_TEMP_FLAG msg))
    (cl:cons ':COMM_ERROR_FLAG (COMM_ERROR_FLAG msg))
    (cl:cons ':BMS_SOC (BMS_SOC msg))
    (cl:cons ':BMS_SOH (BMS_SOH msg))
    (cl:cons ':BMS_Voltage (BMS_Voltage msg))
    (cl:cons ':BMS_Current (BMS_Current msg))
    (cl:cons ':BMS_voltageDiff (BMS_voltageDiff msg))
    (cl:cons ':BMS_tempDiff (BMS_tempDiff msg))
    (cl:cons ':BMS_BCUMode (BMS_BCUMode msg))
    (cl:cons ':BMS_Alarm (BMS_Alarm msg))
    (cl:cons ':M_State (M_State msg))
    (cl:cons ':M_Alarm (M_Alarm msg))
    (cl:cons ':M_Power (M_Power msg))
    (cl:cons ':M_Voltage (M_Voltage msg))
    (cl:cons ':M_Current (M_Current msg))
    (cl:cons ':M_Comm (M_Comm msg))
    (cl:cons ':Relay_Temp (Relay_Temp msg))
    (cl:cons ':PowerConn_Temp (PowerConn_Temp msg))
    (cl:cons ':PreCharge_Temp (PreCharge_Temp msg))
    (cl:cons ':Power_Supply_Temp (Power_Supply_Temp msg))
    (cl:cons ':Battery_In_Volt (Battery_In_Volt msg))
    (cl:cons ':PRE_CHARGE_Volt (PRE_CHARGE_Volt msg))
    (cl:cons ':HW_ESTOP_Volt (HW_ESTOP_Volt msg))
    (cl:cons ':SW_ESTOP_Volt (SW_ESTOP_Volt msg))
    (cl:cons ':Motor_Alarm_Code (Motor_Alarm_Code msg))
    (cl:cons ':Motor_Fun_Fail (Motor_Fun_Fail msg))
    (cl:cons ':M_INIT (M_INIT msg))
    (cl:cons ':M_CW (M_CW msg))
    (cl:cons ':M_CCW (M_CCW msg))
    (cl:cons ':M_STOP (M_STOP msg))
    (cl:cons ':M_BREAK (M_BREAK msg))
    (cl:cons ':M_SETRPM (M_SETRPM msg))
    (cl:cons ':M_GETRPM (M_GETRPM msg))
    (cl:cons ':M_PARAM (M_PARAM msg))
    (cl:cons ':M_ALARM (M_ALARM msg))
    (cl:cons ':PRECHARGE_FUSE (PRECHARGE_FUSE msg))
    (cl:cons ':MOTOR_FUSE (MOTOR_FUSE msg))
    (cl:cons ':LED_STATE (LED_STATE msg))
))

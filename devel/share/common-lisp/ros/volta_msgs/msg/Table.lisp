; Auto-generated. Do not edit!


(cl:in-package volta_msgs-msg)


;//! \htmlinclude Table.msg.html

(cl:defclass <Table> (roslisp-msg-protocol:ros-message)
  ((soc
    :reader soc
    :initarg :soc
    :type cl:float
    :initform 0.0)
   (soh
    :reader soh
    :initarg :soh
    :type cl:float
    :initform 0.0)
   (current
    :reader current
    :initarg :current
    :type cl:float
    :initform 0.0)
   (voltage
    :reader voltage
    :initarg :voltage
    :type cl:float
    :initform 0.0)
   (system_status
    :reader system_status
    :initarg :system_status
    :type cl:fixnum
    :initform 0)
   (relay_temp
    :reader relay_temp
    :initarg :relay_temp
    :type cl:float
    :initform 0.0)
   (powe_in_temp
    :reader powe_in_temp
    :initarg :powe_in_temp
    :type cl:float
    :initform 0.0)
   (motor_volt
    :reader motor_volt
    :initarg :motor_volt
    :type cl:float
    :initform 0.0)
   (battery_volt
    :reader battery_volt
    :initarg :battery_volt
    :type cl:float
    :initform 0.0)
   (hw_estop_volt
    :reader hw_estop_volt
    :initarg :hw_estop_volt
    :type cl:float
    :initform 0.0)
   (motorAlarm
    :reader motorAlarm
    :initarg :motorAlarm
    :type cl:fixnum
    :initform 0)
   (diag_en
    :reader diag_en
    :initarg :diag_en
    :type cl:fixnum
    :initform 0)
   (hw_Estop_state
    :reader hw_Estop_state
    :initarg :hw_Estop_state
    :type cl:fixnum
    :initform 0)
   (sw_Estop_state
    :reader sw_Estop_state
    :initarg :sw_Estop_state
    :type cl:fixnum
    :initform 0)
   (ros_Estop_state
    :reader ros_Estop_state
    :initarg :ros_Estop_state
    :type cl:fixnum
    :initform 0))
)

(cl:defclass Table (<Table>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Table>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Table)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name volta_msgs-msg:<Table> is deprecated: use volta_msgs-msg:Table instead.")))

(cl:ensure-generic-function 'soc-val :lambda-list '(m))
(cl:defmethod soc-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:soc-val is deprecated.  Use volta_msgs-msg:soc instead.")
  (soc m))

(cl:ensure-generic-function 'soh-val :lambda-list '(m))
(cl:defmethod soh-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:soh-val is deprecated.  Use volta_msgs-msg:soh instead.")
  (soh m))

(cl:ensure-generic-function 'current-val :lambda-list '(m))
(cl:defmethod current-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:current-val is deprecated.  Use volta_msgs-msg:current instead.")
  (current m))

(cl:ensure-generic-function 'voltage-val :lambda-list '(m))
(cl:defmethod voltage-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:voltage-val is deprecated.  Use volta_msgs-msg:voltage instead.")
  (voltage m))

(cl:ensure-generic-function 'system_status-val :lambda-list '(m))
(cl:defmethod system_status-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:system_status-val is deprecated.  Use volta_msgs-msg:system_status instead.")
  (system_status m))

(cl:ensure-generic-function 'relay_temp-val :lambda-list '(m))
(cl:defmethod relay_temp-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:relay_temp-val is deprecated.  Use volta_msgs-msg:relay_temp instead.")
  (relay_temp m))

(cl:ensure-generic-function 'powe_in_temp-val :lambda-list '(m))
(cl:defmethod powe_in_temp-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:powe_in_temp-val is deprecated.  Use volta_msgs-msg:powe_in_temp instead.")
  (powe_in_temp m))

(cl:ensure-generic-function 'motor_volt-val :lambda-list '(m))
(cl:defmethod motor_volt-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:motor_volt-val is deprecated.  Use volta_msgs-msg:motor_volt instead.")
  (motor_volt m))

(cl:ensure-generic-function 'battery_volt-val :lambda-list '(m))
(cl:defmethod battery_volt-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:battery_volt-val is deprecated.  Use volta_msgs-msg:battery_volt instead.")
  (battery_volt m))

(cl:ensure-generic-function 'hw_estop_volt-val :lambda-list '(m))
(cl:defmethod hw_estop_volt-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:hw_estop_volt-val is deprecated.  Use volta_msgs-msg:hw_estop_volt instead.")
  (hw_estop_volt m))

(cl:ensure-generic-function 'motorAlarm-val :lambda-list '(m))
(cl:defmethod motorAlarm-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:motorAlarm-val is deprecated.  Use volta_msgs-msg:motorAlarm instead.")
  (motorAlarm m))

(cl:ensure-generic-function 'diag_en-val :lambda-list '(m))
(cl:defmethod diag_en-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:diag_en-val is deprecated.  Use volta_msgs-msg:diag_en instead.")
  (diag_en m))

(cl:ensure-generic-function 'hw_Estop_state-val :lambda-list '(m))
(cl:defmethod hw_Estop_state-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:hw_Estop_state-val is deprecated.  Use volta_msgs-msg:hw_Estop_state instead.")
  (hw_Estop_state m))

(cl:ensure-generic-function 'sw_Estop_state-val :lambda-list '(m))
(cl:defmethod sw_Estop_state-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:sw_Estop_state-val is deprecated.  Use volta_msgs-msg:sw_Estop_state instead.")
  (sw_Estop_state m))

(cl:ensure-generic-function 'ros_Estop_state-val :lambda-list '(m))
(cl:defmethod ros_Estop_state-val ((m <Table>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:ros_Estop_state-val is deprecated.  Use volta_msgs-msg:ros_Estop_state instead.")
  (ros_Estop_state m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Table>) ostream)
  "Serializes a message object of type '<Table>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'soc))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'soh))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'current))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'voltage))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_status)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'relay_temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'powe_in_temp))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'motor_volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'battery_volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'hw_estop_volt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motorAlarm)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'diag_en)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'hw_Estop_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sw_Estop_state)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ros_Estop_state)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Table>) istream)
  "Deserializes a message object of type '<Table>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'soc) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'soh) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'current) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'voltage) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'system_status)) (cl:read-byte istream))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'relay_temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'powe_in_temp) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'motor_volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'battery_volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'hw_estop_volt) (roslisp-utils:decode-single-float-bits bits)))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'motorAlarm)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'diag_en)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'hw_Estop_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'sw_Estop_state)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'ros_Estop_state)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Table>)))
  "Returns string type for a message object of type '<Table>"
  "volta_msgs/Table")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Table)))
  "Returns string type for a message object of type 'Table"
  "volta_msgs/Table")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Table>)))
  "Returns md5sum for a message object of type '<Table>"
  "b849cd8a5229dad6cf68bbda639aeae9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Table)))
  "Returns md5sum for a message object of type 'Table"
  "b849cd8a5229dad6cf68bbda639aeae9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Table>)))
  "Returns full string definition for message of type '<Table>"
  (cl:format cl:nil "float32 soc~%float32 soh~%float32 current~%float32 voltage~%uint8 system_status~%float32 relay_temp~%float32 powe_in_temp~%~%float32 motor_volt~%float32 battery_volt~%float32 hw_estop_volt~%uint8 motorAlarm~%uint8 diag_en~%uint8 hw_Estop_state~%uint8 sw_Estop_state~%uint8 ros_Estop_state~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Table)))
  "Returns full string definition for message of type 'Table"
  (cl:format cl:nil "float32 soc~%float32 soh~%float32 current~%float32 voltage~%uint8 system_status~%float32 relay_temp~%float32 powe_in_temp~%~%float32 motor_volt~%float32 battery_volt~%float32 hw_estop_volt~%uint8 motorAlarm~%uint8 diag_en~%uint8 hw_Estop_state~%uint8 sw_Estop_state~%uint8 ros_Estop_state~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Table>))
  (cl:+ 0
     4
     4
     4
     4
     1
     4
     4
     4
     4
     4
     1
     1
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Table>))
  "Converts a ROS message object to a list"
  (cl:list 'Table
    (cl:cons ':soc (soc msg))
    (cl:cons ':soh (soh msg))
    (cl:cons ':current (current msg))
    (cl:cons ':voltage (voltage msg))
    (cl:cons ':system_status (system_status msg))
    (cl:cons ':relay_temp (relay_temp msg))
    (cl:cons ':powe_in_temp (powe_in_temp msg))
    (cl:cons ':motor_volt (motor_volt msg))
    (cl:cons ':battery_volt (battery_volt msg))
    (cl:cons ':hw_estop_volt (hw_estop_volt msg))
    (cl:cons ':motorAlarm (motorAlarm msg))
    (cl:cons ':diag_en (diag_en msg))
    (cl:cons ':hw_Estop_state (hw_Estop_state msg))
    (cl:cons ':sw_Estop_state (sw_Estop_state msg))
    (cl:cons ':ros_Estop_state (ros_Estop_state msg))
))

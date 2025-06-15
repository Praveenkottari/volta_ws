; Auto-generated. Do not edit!


(cl:in-package volta_msgs-msg)


;//! \htmlinclude BMS.msg.html

(cl:defclass <BMS> (roslisp-msg-protocol:ros-message)
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
    :initform 0.0))
)

(cl:defclass BMS (<BMS>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <BMS>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'BMS)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name volta_msgs-msg:<BMS> is deprecated: use volta_msgs-msg:BMS instead.")))

(cl:ensure-generic-function 'soc-val :lambda-list '(m))
(cl:defmethod soc-val ((m <BMS>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:soc-val is deprecated.  Use volta_msgs-msg:soc instead.")
  (soc m))

(cl:ensure-generic-function 'soh-val :lambda-list '(m))
(cl:defmethod soh-val ((m <BMS>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:soh-val is deprecated.  Use volta_msgs-msg:soh instead.")
  (soh m))

(cl:ensure-generic-function 'current-val :lambda-list '(m))
(cl:defmethod current-val ((m <BMS>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:current-val is deprecated.  Use volta_msgs-msg:current instead.")
  (current m))

(cl:ensure-generic-function 'voltage-val :lambda-list '(m))
(cl:defmethod voltage-val ((m <BMS>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:voltage-val is deprecated.  Use volta_msgs-msg:voltage instead.")
  (voltage m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <BMS>) ostream)
  "Serializes a message object of type '<BMS>"
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <BMS>) istream)
  "Deserializes a message object of type '<BMS>"
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<BMS>)))
  "Returns string type for a message object of type '<BMS>"
  "volta_msgs/BMS")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'BMS)))
  "Returns string type for a message object of type 'BMS"
  "volta_msgs/BMS")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<BMS>)))
  "Returns md5sum for a message object of type '<BMS>"
  "e5d0e1c804606ac58f230d85fba314ed")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'BMS)))
  "Returns md5sum for a message object of type 'BMS"
  "e5d0e1c804606ac58f230d85fba314ed")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<BMS>)))
  "Returns full string definition for message of type '<BMS>"
  (cl:format cl:nil "float32 soc~%float32 soh~%float32 current~%float32 voltage~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'BMS)))
  "Returns full string definition for message of type 'BMS"
  (cl:format cl:nil "float32 soc~%float32 soh~%float32 current~%float32 voltage~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <BMS>))
  (cl:+ 0
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <BMS>))
  "Converts a ROS message object to a list"
  (cl:list 'BMS
    (cl:cons ':soc (soc msg))
    (cl:cons ':soh (soh msg))
    (cl:cons ':current (current msg))
    (cl:cons ':voltage (voltage msg))
))

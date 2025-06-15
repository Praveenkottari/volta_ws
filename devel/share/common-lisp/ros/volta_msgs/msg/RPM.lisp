; Auto-generated. Do not edit!


(cl:in-package volta_msgs-msg)


;//! \htmlinclude RPM.msg.html

(cl:defclass <RPM> (roslisp-msg-protocol:ros-message)
  ((left
    :reader left
    :initarg :left
    :type cl:fixnum
    :initform 0)
   (right
    :reader right
    :initarg :right
    :type cl:fixnum
    :initform 0))
)

(cl:defclass RPM (<RPM>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RPM>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RPM)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name volta_msgs-msg:<RPM> is deprecated: use volta_msgs-msg:RPM instead.")))

(cl:ensure-generic-function 'left-val :lambda-list '(m))
(cl:defmethod left-val ((m <RPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:left-val is deprecated.  Use volta_msgs-msg:left instead.")
  (left m))

(cl:ensure-generic-function 'right-val :lambda-list '(m))
(cl:defmethod right-val ((m <RPM>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader volta_msgs-msg:right-val is deprecated.  Use volta_msgs-msg:right instead.")
  (right m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RPM>) ostream)
  "Serializes a message object of type '<RPM>"
  (cl:let* ((signed (cl:slot-value msg 'left)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'right)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RPM>) istream)
  "Deserializes a message object of type '<RPM>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'left) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'right) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RPM>)))
  "Returns string type for a message object of type '<RPM>"
  "volta_msgs/RPM")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RPM)))
  "Returns string type for a message object of type 'RPM"
  "volta_msgs/RPM")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RPM>)))
  "Returns md5sum for a message object of type '<RPM>"
  "09d1b2323a1aeae9343e81809a820b60")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RPM)))
  "Returns md5sum for a message object of type 'RPM"
  "09d1b2323a1aeae9343e81809a820b60")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RPM>)))
  "Returns full string definition for message of type '<RPM>"
  (cl:format cl:nil "int16 left~%int16 right~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RPM)))
  "Returns full string definition for message of type 'RPM"
  (cl:format cl:nil "int16 left~%int16 right~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RPM>))
  (cl:+ 0
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RPM>))
  "Converts a ROS message object to a list"
  (cl:list 'RPM
    (cl:cons ':left (left msg))
    (cl:cons ':right (right msg))
))

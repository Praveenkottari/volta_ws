// Auto-generated. Do not edit!

// (in-package volta_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Table {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.soc = null;
      this.soh = null;
      this.current = null;
      this.voltage = null;
      this.system_status = null;
      this.relay_temp = null;
      this.powe_in_temp = null;
      this.motor_volt = null;
      this.battery_volt = null;
      this.hw_estop_volt = null;
      this.motorAlarm = null;
      this.diag_en = null;
      this.hw_Estop_state = null;
      this.sw_Estop_state = null;
      this.ros_Estop_state = null;
    }
    else {
      if (initObj.hasOwnProperty('soc')) {
        this.soc = initObj.soc
      }
      else {
        this.soc = 0.0;
      }
      if (initObj.hasOwnProperty('soh')) {
        this.soh = initObj.soh
      }
      else {
        this.soh = 0.0;
      }
      if (initObj.hasOwnProperty('current')) {
        this.current = initObj.current
      }
      else {
        this.current = 0.0;
      }
      if (initObj.hasOwnProperty('voltage')) {
        this.voltage = initObj.voltage
      }
      else {
        this.voltage = 0.0;
      }
      if (initObj.hasOwnProperty('system_status')) {
        this.system_status = initObj.system_status
      }
      else {
        this.system_status = 0;
      }
      if (initObj.hasOwnProperty('relay_temp')) {
        this.relay_temp = initObj.relay_temp
      }
      else {
        this.relay_temp = 0.0;
      }
      if (initObj.hasOwnProperty('powe_in_temp')) {
        this.powe_in_temp = initObj.powe_in_temp
      }
      else {
        this.powe_in_temp = 0.0;
      }
      if (initObj.hasOwnProperty('motor_volt')) {
        this.motor_volt = initObj.motor_volt
      }
      else {
        this.motor_volt = 0.0;
      }
      if (initObj.hasOwnProperty('battery_volt')) {
        this.battery_volt = initObj.battery_volt
      }
      else {
        this.battery_volt = 0.0;
      }
      if (initObj.hasOwnProperty('hw_estop_volt')) {
        this.hw_estop_volt = initObj.hw_estop_volt
      }
      else {
        this.hw_estop_volt = 0.0;
      }
      if (initObj.hasOwnProperty('motorAlarm')) {
        this.motorAlarm = initObj.motorAlarm
      }
      else {
        this.motorAlarm = 0;
      }
      if (initObj.hasOwnProperty('diag_en')) {
        this.diag_en = initObj.diag_en
      }
      else {
        this.diag_en = 0;
      }
      if (initObj.hasOwnProperty('hw_Estop_state')) {
        this.hw_Estop_state = initObj.hw_Estop_state
      }
      else {
        this.hw_Estop_state = 0;
      }
      if (initObj.hasOwnProperty('sw_Estop_state')) {
        this.sw_Estop_state = initObj.sw_Estop_state
      }
      else {
        this.sw_Estop_state = 0;
      }
      if (initObj.hasOwnProperty('ros_Estop_state')) {
        this.ros_Estop_state = initObj.ros_Estop_state
      }
      else {
        this.ros_Estop_state = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Table
    // Serialize message field [soc]
    bufferOffset = _serializer.float32(obj.soc, buffer, bufferOffset);
    // Serialize message field [soh]
    bufferOffset = _serializer.float32(obj.soh, buffer, bufferOffset);
    // Serialize message field [current]
    bufferOffset = _serializer.float32(obj.current, buffer, bufferOffset);
    // Serialize message field [voltage]
    bufferOffset = _serializer.float32(obj.voltage, buffer, bufferOffset);
    // Serialize message field [system_status]
    bufferOffset = _serializer.uint8(obj.system_status, buffer, bufferOffset);
    // Serialize message field [relay_temp]
    bufferOffset = _serializer.float32(obj.relay_temp, buffer, bufferOffset);
    // Serialize message field [powe_in_temp]
    bufferOffset = _serializer.float32(obj.powe_in_temp, buffer, bufferOffset);
    // Serialize message field [motor_volt]
    bufferOffset = _serializer.float32(obj.motor_volt, buffer, bufferOffset);
    // Serialize message field [battery_volt]
    bufferOffset = _serializer.float32(obj.battery_volt, buffer, bufferOffset);
    // Serialize message field [hw_estop_volt]
    bufferOffset = _serializer.float32(obj.hw_estop_volt, buffer, bufferOffset);
    // Serialize message field [motorAlarm]
    bufferOffset = _serializer.uint8(obj.motorAlarm, buffer, bufferOffset);
    // Serialize message field [diag_en]
    bufferOffset = _serializer.uint8(obj.diag_en, buffer, bufferOffset);
    // Serialize message field [hw_Estop_state]
    bufferOffset = _serializer.uint8(obj.hw_Estop_state, buffer, bufferOffset);
    // Serialize message field [sw_Estop_state]
    bufferOffset = _serializer.uint8(obj.sw_Estop_state, buffer, bufferOffset);
    // Serialize message field [ros_Estop_state]
    bufferOffset = _serializer.uint8(obj.ros_Estop_state, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Table
    let len;
    let data = new Table(null);
    // Deserialize message field [soc]
    data.soc = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [soh]
    data.soh = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [current]
    data.current = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [voltage]
    data.voltage = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [system_status]
    data.system_status = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [relay_temp]
    data.relay_temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [powe_in_temp]
    data.powe_in_temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [motor_volt]
    data.motor_volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [battery_volt]
    data.battery_volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [hw_estop_volt]
    data.hw_estop_volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [motorAlarm]
    data.motorAlarm = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [diag_en]
    data.diag_en = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [hw_Estop_state]
    data.hw_Estop_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [sw_Estop_state]
    data.sw_Estop_state = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [ros_Estop_state]
    data.ros_Estop_state = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 42;
  }

  static datatype() {
    // Returns string type for a message object
    return 'volta_msgs/Table';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b849cd8a5229dad6cf68bbda639aeae9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 soc
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Table(null);
    if (msg.soc !== undefined) {
      resolved.soc = msg.soc;
    }
    else {
      resolved.soc = 0.0
    }

    if (msg.soh !== undefined) {
      resolved.soh = msg.soh;
    }
    else {
      resolved.soh = 0.0
    }

    if (msg.current !== undefined) {
      resolved.current = msg.current;
    }
    else {
      resolved.current = 0.0
    }

    if (msg.voltage !== undefined) {
      resolved.voltage = msg.voltage;
    }
    else {
      resolved.voltage = 0.0
    }

    if (msg.system_status !== undefined) {
      resolved.system_status = msg.system_status;
    }
    else {
      resolved.system_status = 0
    }

    if (msg.relay_temp !== undefined) {
      resolved.relay_temp = msg.relay_temp;
    }
    else {
      resolved.relay_temp = 0.0
    }

    if (msg.powe_in_temp !== undefined) {
      resolved.powe_in_temp = msg.powe_in_temp;
    }
    else {
      resolved.powe_in_temp = 0.0
    }

    if (msg.motor_volt !== undefined) {
      resolved.motor_volt = msg.motor_volt;
    }
    else {
      resolved.motor_volt = 0.0
    }

    if (msg.battery_volt !== undefined) {
      resolved.battery_volt = msg.battery_volt;
    }
    else {
      resolved.battery_volt = 0.0
    }

    if (msg.hw_estop_volt !== undefined) {
      resolved.hw_estop_volt = msg.hw_estop_volt;
    }
    else {
      resolved.hw_estop_volt = 0.0
    }

    if (msg.motorAlarm !== undefined) {
      resolved.motorAlarm = msg.motorAlarm;
    }
    else {
      resolved.motorAlarm = 0
    }

    if (msg.diag_en !== undefined) {
      resolved.diag_en = msg.diag_en;
    }
    else {
      resolved.diag_en = 0
    }

    if (msg.hw_Estop_state !== undefined) {
      resolved.hw_Estop_state = msg.hw_Estop_state;
    }
    else {
      resolved.hw_Estop_state = 0
    }

    if (msg.sw_Estop_state !== undefined) {
      resolved.sw_Estop_state = msg.sw_Estop_state;
    }
    else {
      resolved.sw_Estop_state = 0
    }

    if (msg.ros_Estop_state !== undefined) {
      resolved.ros_Estop_state = msg.ros_Estop_state;
    }
    else {
      resolved.ros_Estop_state = 0
    }

    return resolved;
    }
};

module.exports = Table;

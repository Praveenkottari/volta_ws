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

class BMS {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.soc = null;
      this.soh = null;
      this.current = null;
      this.voltage = null;
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
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type BMS
    // Serialize message field [soc]
    bufferOffset = _serializer.float32(obj.soc, buffer, bufferOffset);
    // Serialize message field [soh]
    bufferOffset = _serializer.float32(obj.soh, buffer, bufferOffset);
    // Serialize message field [current]
    bufferOffset = _serializer.float32(obj.current, buffer, bufferOffset);
    // Serialize message field [voltage]
    bufferOffset = _serializer.float32(obj.voltage, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type BMS
    let len;
    let data = new BMS(null);
    // Deserialize message field [soc]
    data.soc = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [soh]
    data.soh = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [current]
    data.current = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [voltage]
    data.voltage = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'volta_msgs/BMS';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e5d0e1c804606ac58f230d85fba314ed';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 soc
    float32 soh
    float32 current
    float32 voltage
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new BMS(null);
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

    return resolved;
    }
};

module.exports = BMS;

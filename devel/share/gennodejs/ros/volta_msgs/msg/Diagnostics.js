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

class Diagnostics {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.LOWBATTERY_FLAG = null;
      this.CHARGING_FLAG = null;
      this.MOTOR_ALARM_FLAG = null;
      this.BMS_ALARM_FLAG = null;
      this.HIGH_TEMP_FLAG = null;
      this.COMM_ERROR_FLAG = null;
      this.BMS_SOC = null;
      this.BMS_SOH = null;
      this.BMS_Voltage = null;
      this.BMS_Current = null;
      this.BMS_voltageDiff = null;
      this.BMS_tempDiff = null;
      this.BMS_BCUMode = null;
      this.BMS_Alarm = null;
      this.M_State = null;
      this.M_Alarm = null;
      this.M_Power = null;
      this.M_Voltage = null;
      this.M_Current = null;
      this.M_Comm = null;
      this.Relay_Temp = null;
      this.PowerConn_Temp = null;
      this.PreCharge_Temp = null;
      this.Power_Supply_Temp = null;
      this.Battery_In_Volt = null;
      this.PRE_CHARGE_Volt = null;
      this.HW_ESTOP_Volt = null;
      this.SW_ESTOP_Volt = null;
      this.Motor_Alarm_Code = null;
      this.Motor_Fun_Fail = null;
      this.M_INIT = null;
      this.M_CW = null;
      this.M_CCW = null;
      this.M_STOP = null;
      this.M_BREAK = null;
      this.M_SETRPM = null;
      this.M_GETRPM = null;
      this.M_PARAM = null;
      this.M_ALARM = null;
      this.PRECHARGE_FUSE = null;
      this.MOTOR_FUSE = null;
      this.LED_STATE = null;
    }
    else {
      if (initObj.hasOwnProperty('LOWBATTERY_FLAG')) {
        this.LOWBATTERY_FLAG = initObj.LOWBATTERY_FLAG
      }
      else {
        this.LOWBATTERY_FLAG = false;
      }
      if (initObj.hasOwnProperty('CHARGING_FLAG')) {
        this.CHARGING_FLAG = initObj.CHARGING_FLAG
      }
      else {
        this.CHARGING_FLAG = false;
      }
      if (initObj.hasOwnProperty('MOTOR_ALARM_FLAG')) {
        this.MOTOR_ALARM_FLAG = initObj.MOTOR_ALARM_FLAG
      }
      else {
        this.MOTOR_ALARM_FLAG = false;
      }
      if (initObj.hasOwnProperty('BMS_ALARM_FLAG')) {
        this.BMS_ALARM_FLAG = initObj.BMS_ALARM_FLAG
      }
      else {
        this.BMS_ALARM_FLAG = false;
      }
      if (initObj.hasOwnProperty('HIGH_TEMP_FLAG')) {
        this.HIGH_TEMP_FLAG = initObj.HIGH_TEMP_FLAG
      }
      else {
        this.HIGH_TEMP_FLAG = false;
      }
      if (initObj.hasOwnProperty('COMM_ERROR_FLAG')) {
        this.COMM_ERROR_FLAG = initObj.COMM_ERROR_FLAG
      }
      else {
        this.COMM_ERROR_FLAG = false;
      }
      if (initObj.hasOwnProperty('BMS_SOC')) {
        this.BMS_SOC = initObj.BMS_SOC
      }
      else {
        this.BMS_SOC = 0;
      }
      if (initObj.hasOwnProperty('BMS_SOH')) {
        this.BMS_SOH = initObj.BMS_SOH
      }
      else {
        this.BMS_SOH = 0;
      }
      if (initObj.hasOwnProperty('BMS_Voltage')) {
        this.BMS_Voltage = initObj.BMS_Voltage
      }
      else {
        this.BMS_Voltage = 0;
      }
      if (initObj.hasOwnProperty('BMS_Current')) {
        this.BMS_Current = initObj.BMS_Current
      }
      else {
        this.BMS_Current = 0;
      }
      if (initObj.hasOwnProperty('BMS_voltageDiff')) {
        this.BMS_voltageDiff = initObj.BMS_voltageDiff
      }
      else {
        this.BMS_voltageDiff = 0;
      }
      if (initObj.hasOwnProperty('BMS_tempDiff')) {
        this.BMS_tempDiff = initObj.BMS_tempDiff
      }
      else {
        this.BMS_tempDiff = 0;
      }
      if (initObj.hasOwnProperty('BMS_BCUMode')) {
        this.BMS_BCUMode = initObj.BMS_BCUMode
      }
      else {
        this.BMS_BCUMode = 0;
      }
      if (initObj.hasOwnProperty('BMS_Alarm')) {
        this.BMS_Alarm = initObj.BMS_Alarm
      }
      else {
        this.BMS_Alarm = 0;
      }
      if (initObj.hasOwnProperty('M_State')) {
        this.M_State = initObj.M_State
      }
      else {
        this.M_State = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_Alarm')) {
        this.M_Alarm = initObj.M_Alarm
      }
      else {
        this.M_Alarm = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_Power')) {
        this.M_Power = initObj.M_Power
      }
      else {
        this.M_Power = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_Voltage')) {
        this.M_Voltage = initObj.M_Voltage
      }
      else {
        this.M_Voltage = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_Current')) {
        this.M_Current = initObj.M_Current
      }
      else {
        this.M_Current = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_Comm')) {
        this.M_Comm = initObj.M_Comm
      }
      else {
        this.M_Comm = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('Relay_Temp')) {
        this.Relay_Temp = initObj.Relay_Temp
      }
      else {
        this.Relay_Temp = 0.0;
      }
      if (initObj.hasOwnProperty('PowerConn_Temp')) {
        this.PowerConn_Temp = initObj.PowerConn_Temp
      }
      else {
        this.PowerConn_Temp = 0.0;
      }
      if (initObj.hasOwnProperty('PreCharge_Temp')) {
        this.PreCharge_Temp = initObj.PreCharge_Temp
      }
      else {
        this.PreCharge_Temp = 0.0;
      }
      if (initObj.hasOwnProperty('Power_Supply_Temp')) {
        this.Power_Supply_Temp = initObj.Power_Supply_Temp
      }
      else {
        this.Power_Supply_Temp = 0.0;
      }
      if (initObj.hasOwnProperty('Battery_In_Volt')) {
        this.Battery_In_Volt = initObj.Battery_In_Volt
      }
      else {
        this.Battery_In_Volt = 0.0;
      }
      if (initObj.hasOwnProperty('PRE_CHARGE_Volt')) {
        this.PRE_CHARGE_Volt = initObj.PRE_CHARGE_Volt
      }
      else {
        this.PRE_CHARGE_Volt = 0.0;
      }
      if (initObj.hasOwnProperty('HW_ESTOP_Volt')) {
        this.HW_ESTOP_Volt = initObj.HW_ESTOP_Volt
      }
      else {
        this.HW_ESTOP_Volt = 0.0;
      }
      if (initObj.hasOwnProperty('SW_ESTOP_Volt')) {
        this.SW_ESTOP_Volt = initObj.SW_ESTOP_Volt
      }
      else {
        this.SW_ESTOP_Volt = 0.0;
      }
      if (initObj.hasOwnProperty('Motor_Alarm_Code')) {
        this.Motor_Alarm_Code = initObj.Motor_Alarm_Code
      }
      else {
        this.Motor_Alarm_Code = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('Motor_Fun_Fail')) {
        this.Motor_Fun_Fail = initObj.Motor_Fun_Fail
      }
      else {
        this.Motor_Fun_Fail = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_INIT')) {
        this.M_INIT = initObj.M_INIT
      }
      else {
        this.M_INIT = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_CW')) {
        this.M_CW = initObj.M_CW
      }
      else {
        this.M_CW = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_CCW')) {
        this.M_CCW = initObj.M_CCW
      }
      else {
        this.M_CCW = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_STOP')) {
        this.M_STOP = initObj.M_STOP
      }
      else {
        this.M_STOP = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_BREAK')) {
        this.M_BREAK = initObj.M_BREAK
      }
      else {
        this.M_BREAK = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_SETRPM')) {
        this.M_SETRPM = initObj.M_SETRPM
      }
      else {
        this.M_SETRPM = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_GETRPM')) {
        this.M_GETRPM = initObj.M_GETRPM
      }
      else {
        this.M_GETRPM = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_PARAM')) {
        this.M_PARAM = initObj.M_PARAM
      }
      else {
        this.M_PARAM = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('M_ALARM')) {
        this.M_ALARM = initObj.M_ALARM
      }
      else {
        this.M_ALARM = new Array(2).fill(0);
      }
      if (initObj.hasOwnProperty('PRECHARGE_FUSE')) {
        this.PRECHARGE_FUSE = initObj.PRECHARGE_FUSE
      }
      else {
        this.PRECHARGE_FUSE = 0;
      }
      if (initObj.hasOwnProperty('MOTOR_FUSE')) {
        this.MOTOR_FUSE = initObj.MOTOR_FUSE
      }
      else {
        this.MOTOR_FUSE = 0;
      }
      if (initObj.hasOwnProperty('LED_STATE')) {
        this.LED_STATE = initObj.LED_STATE
      }
      else {
        this.LED_STATE = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Diagnostics
    // Serialize message field [LOWBATTERY_FLAG]
    bufferOffset = _serializer.bool(obj.LOWBATTERY_FLAG, buffer, bufferOffset);
    // Serialize message field [CHARGING_FLAG]
    bufferOffset = _serializer.bool(obj.CHARGING_FLAG, buffer, bufferOffset);
    // Serialize message field [MOTOR_ALARM_FLAG]
    bufferOffset = _serializer.bool(obj.MOTOR_ALARM_FLAG, buffer, bufferOffset);
    // Serialize message field [BMS_ALARM_FLAG]
    bufferOffset = _serializer.bool(obj.BMS_ALARM_FLAG, buffer, bufferOffset);
    // Serialize message field [HIGH_TEMP_FLAG]
    bufferOffset = _serializer.bool(obj.HIGH_TEMP_FLAG, buffer, bufferOffset);
    // Serialize message field [COMM_ERROR_FLAG]
    bufferOffset = _serializer.bool(obj.COMM_ERROR_FLAG, buffer, bufferOffset);
    // Serialize message field [BMS_SOC]
    bufferOffset = _serializer.uint16(obj.BMS_SOC, buffer, bufferOffset);
    // Serialize message field [BMS_SOH]
    bufferOffset = _serializer.uint16(obj.BMS_SOH, buffer, bufferOffset);
    // Serialize message field [BMS_Voltage]
    bufferOffset = _serializer.uint16(obj.BMS_Voltage, buffer, bufferOffset);
    // Serialize message field [BMS_Current]
    bufferOffset = _serializer.int16(obj.BMS_Current, buffer, bufferOffset);
    // Serialize message field [BMS_voltageDiff]
    bufferOffset = _serializer.uint16(obj.BMS_voltageDiff, buffer, bufferOffset);
    // Serialize message field [BMS_tempDiff]
    bufferOffset = _serializer.uint16(obj.BMS_tempDiff, buffer, bufferOffset);
    // Serialize message field [BMS_BCUMode]
    bufferOffset = _serializer.uint16(obj.BMS_BCUMode, buffer, bufferOffset);
    // Serialize message field [BMS_Alarm]
    bufferOffset = _serializer.uint16(obj.BMS_Alarm, buffer, bufferOffset);
    // Check that the constant length array field [M_State] has the right length
    if (obj.M_State.length !== 2) {
      throw new Error('Unable to serialize array field M_State - length must be 2')
    }
    // Serialize message field [M_State]
    bufferOffset = _arraySerializer.uint16(obj.M_State, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_Alarm] has the right length
    if (obj.M_Alarm.length !== 2) {
      throw new Error('Unable to serialize array field M_Alarm - length must be 2')
    }
    // Serialize message field [M_Alarm]
    bufferOffset = _arraySerializer.uint16(obj.M_Alarm, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_Power] has the right length
    if (obj.M_Power.length !== 2) {
      throw new Error('Unable to serialize array field M_Power - length must be 2')
    }
    // Serialize message field [M_Power]
    bufferOffset = _arraySerializer.uint16(obj.M_Power, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_Voltage] has the right length
    if (obj.M_Voltage.length !== 2) {
      throw new Error('Unable to serialize array field M_Voltage - length must be 2')
    }
    // Serialize message field [M_Voltage]
    bufferOffset = _arraySerializer.uint16(obj.M_Voltage, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_Current] has the right length
    if (obj.M_Current.length !== 2) {
      throw new Error('Unable to serialize array field M_Current - length must be 2')
    }
    // Serialize message field [M_Current]
    bufferOffset = _arraySerializer.uint16(obj.M_Current, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_Comm] has the right length
    if (obj.M_Comm.length !== 2) {
      throw new Error('Unable to serialize array field M_Comm - length must be 2')
    }
    // Serialize message field [M_Comm]
    bufferOffset = _arraySerializer.uint8(obj.M_Comm, buffer, bufferOffset, 2);
    // Serialize message field [Relay_Temp]
    bufferOffset = _serializer.float32(obj.Relay_Temp, buffer, bufferOffset);
    // Serialize message field [PowerConn_Temp]
    bufferOffset = _serializer.float32(obj.PowerConn_Temp, buffer, bufferOffset);
    // Serialize message field [PreCharge_Temp]
    bufferOffset = _serializer.float32(obj.PreCharge_Temp, buffer, bufferOffset);
    // Serialize message field [Power_Supply_Temp]
    bufferOffset = _serializer.float32(obj.Power_Supply_Temp, buffer, bufferOffset);
    // Serialize message field [Battery_In_Volt]
    bufferOffset = _serializer.float32(obj.Battery_In_Volt, buffer, bufferOffset);
    // Serialize message field [PRE_CHARGE_Volt]
    bufferOffset = _serializer.float32(obj.PRE_CHARGE_Volt, buffer, bufferOffset);
    // Serialize message field [HW_ESTOP_Volt]
    bufferOffset = _serializer.float32(obj.HW_ESTOP_Volt, buffer, bufferOffset);
    // Serialize message field [SW_ESTOP_Volt]
    bufferOffset = _serializer.float32(obj.SW_ESTOP_Volt, buffer, bufferOffset);
    // Check that the constant length array field [Motor_Alarm_Code] has the right length
    if (obj.Motor_Alarm_Code.length !== 2) {
      throw new Error('Unable to serialize array field Motor_Alarm_Code - length must be 2')
    }
    // Serialize message field [Motor_Alarm_Code]
    bufferOffset = _arraySerializer.uint16(obj.Motor_Alarm_Code, buffer, bufferOffset, 2);
    // Check that the constant length array field [Motor_Fun_Fail] has the right length
    if (obj.Motor_Fun_Fail.length !== 2) {
      throw new Error('Unable to serialize array field Motor_Fun_Fail - length must be 2')
    }
    // Serialize message field [Motor_Fun_Fail]
    bufferOffset = _arraySerializer.uint32(obj.Motor_Fun_Fail, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_INIT] has the right length
    if (obj.M_INIT.length !== 2) {
      throw new Error('Unable to serialize array field M_INIT - length must be 2')
    }
    // Serialize message field [M_INIT]
    bufferOffset = _arraySerializer.uint16(obj.M_INIT, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_CW] has the right length
    if (obj.M_CW.length !== 2) {
      throw new Error('Unable to serialize array field M_CW - length must be 2')
    }
    // Serialize message field [M_CW]
    bufferOffset = _arraySerializer.uint16(obj.M_CW, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_CCW] has the right length
    if (obj.M_CCW.length !== 2) {
      throw new Error('Unable to serialize array field M_CCW - length must be 2')
    }
    // Serialize message field [M_CCW]
    bufferOffset = _arraySerializer.uint16(obj.M_CCW, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_STOP] has the right length
    if (obj.M_STOP.length !== 2) {
      throw new Error('Unable to serialize array field M_STOP - length must be 2')
    }
    // Serialize message field [M_STOP]
    bufferOffset = _arraySerializer.uint16(obj.M_STOP, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_BREAK] has the right length
    if (obj.M_BREAK.length !== 2) {
      throw new Error('Unable to serialize array field M_BREAK - length must be 2')
    }
    // Serialize message field [M_BREAK]
    bufferOffset = _arraySerializer.uint16(obj.M_BREAK, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_SETRPM] has the right length
    if (obj.M_SETRPM.length !== 2) {
      throw new Error('Unable to serialize array field M_SETRPM - length must be 2')
    }
    // Serialize message field [M_SETRPM]
    bufferOffset = _arraySerializer.uint16(obj.M_SETRPM, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_GETRPM] has the right length
    if (obj.M_GETRPM.length !== 2) {
      throw new Error('Unable to serialize array field M_GETRPM - length must be 2')
    }
    // Serialize message field [M_GETRPM]
    bufferOffset = _arraySerializer.uint16(obj.M_GETRPM, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_PARAM] has the right length
    if (obj.M_PARAM.length !== 2) {
      throw new Error('Unable to serialize array field M_PARAM - length must be 2')
    }
    // Serialize message field [M_PARAM]
    bufferOffset = _arraySerializer.uint16(obj.M_PARAM, buffer, bufferOffset, 2);
    // Check that the constant length array field [M_ALARM] has the right length
    if (obj.M_ALARM.length !== 2) {
      throw new Error('Unable to serialize array field M_ALARM - length must be 2')
    }
    // Serialize message field [M_ALARM]
    bufferOffset = _arraySerializer.uint16(obj.M_ALARM, buffer, bufferOffset, 2);
    // Serialize message field [PRECHARGE_FUSE]
    bufferOffset = _serializer.uint8(obj.PRECHARGE_FUSE, buffer, bufferOffset);
    // Serialize message field [MOTOR_FUSE]
    bufferOffset = _serializer.uint8(obj.MOTOR_FUSE, buffer, bufferOffset);
    // Serialize message field [LED_STATE]
    bufferOffset = _serializer.uint8(obj.LED_STATE, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Diagnostics
    let len;
    let data = new Diagnostics(null);
    // Deserialize message field [LOWBATTERY_FLAG]
    data.LOWBATTERY_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [CHARGING_FLAG]
    data.CHARGING_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [MOTOR_ALARM_FLAG]
    data.MOTOR_ALARM_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [BMS_ALARM_FLAG]
    data.BMS_ALARM_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [HIGH_TEMP_FLAG]
    data.HIGH_TEMP_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [COMM_ERROR_FLAG]
    data.COMM_ERROR_FLAG = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [BMS_SOC]
    data.BMS_SOC = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_SOH]
    data.BMS_SOH = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_Voltage]
    data.BMS_Voltage = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_Current]
    data.BMS_Current = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [BMS_voltageDiff]
    data.BMS_voltageDiff = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_tempDiff]
    data.BMS_tempDiff = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_BCUMode]
    data.BMS_BCUMode = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [BMS_Alarm]
    data.BMS_Alarm = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [M_State]
    data.M_State = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_Alarm]
    data.M_Alarm = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_Power]
    data.M_Power = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_Voltage]
    data.M_Voltage = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_Current]
    data.M_Current = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_Comm]
    data.M_Comm = _arrayDeserializer.uint8(buffer, bufferOffset, 2)
    // Deserialize message field [Relay_Temp]
    data.Relay_Temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [PowerConn_Temp]
    data.PowerConn_Temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [PreCharge_Temp]
    data.PreCharge_Temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Power_Supply_Temp]
    data.Power_Supply_Temp = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Battery_In_Volt]
    data.Battery_In_Volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [PRE_CHARGE_Volt]
    data.PRE_CHARGE_Volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [HW_ESTOP_Volt]
    data.HW_ESTOP_Volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [SW_ESTOP_Volt]
    data.SW_ESTOP_Volt = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Motor_Alarm_Code]
    data.Motor_Alarm_Code = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [Motor_Fun_Fail]
    data.Motor_Fun_Fail = _arrayDeserializer.uint32(buffer, bufferOffset, 2)
    // Deserialize message field [M_INIT]
    data.M_INIT = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_CW]
    data.M_CW = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_CCW]
    data.M_CCW = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_STOP]
    data.M_STOP = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_BREAK]
    data.M_BREAK = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_SETRPM]
    data.M_SETRPM = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_GETRPM]
    data.M_GETRPM = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_PARAM]
    data.M_PARAM = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [M_ALARM]
    data.M_ALARM = _arrayDeserializer.uint16(buffer, bufferOffset, 2)
    // Deserialize message field [PRECHARGE_FUSE]
    data.PRECHARGE_FUSE = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [MOTOR_FUSE]
    data.MOTOR_FUSE = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [LED_STATE]
    data.LED_STATE = _deserializer.uint8(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 127;
  }

  static datatype() {
    // Returns string type for a message object
    return 'volta_msgs/Diagnostics';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '724ceb6ccb5d72d3af9209169201c96c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool LOWBATTERY_FLAG
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Diagnostics(null);
    if (msg.LOWBATTERY_FLAG !== undefined) {
      resolved.LOWBATTERY_FLAG = msg.LOWBATTERY_FLAG;
    }
    else {
      resolved.LOWBATTERY_FLAG = false
    }

    if (msg.CHARGING_FLAG !== undefined) {
      resolved.CHARGING_FLAG = msg.CHARGING_FLAG;
    }
    else {
      resolved.CHARGING_FLAG = false
    }

    if (msg.MOTOR_ALARM_FLAG !== undefined) {
      resolved.MOTOR_ALARM_FLAG = msg.MOTOR_ALARM_FLAG;
    }
    else {
      resolved.MOTOR_ALARM_FLAG = false
    }

    if (msg.BMS_ALARM_FLAG !== undefined) {
      resolved.BMS_ALARM_FLAG = msg.BMS_ALARM_FLAG;
    }
    else {
      resolved.BMS_ALARM_FLAG = false
    }

    if (msg.HIGH_TEMP_FLAG !== undefined) {
      resolved.HIGH_TEMP_FLAG = msg.HIGH_TEMP_FLAG;
    }
    else {
      resolved.HIGH_TEMP_FLAG = false
    }

    if (msg.COMM_ERROR_FLAG !== undefined) {
      resolved.COMM_ERROR_FLAG = msg.COMM_ERROR_FLAG;
    }
    else {
      resolved.COMM_ERROR_FLAG = false
    }

    if (msg.BMS_SOC !== undefined) {
      resolved.BMS_SOC = msg.BMS_SOC;
    }
    else {
      resolved.BMS_SOC = 0
    }

    if (msg.BMS_SOH !== undefined) {
      resolved.BMS_SOH = msg.BMS_SOH;
    }
    else {
      resolved.BMS_SOH = 0
    }

    if (msg.BMS_Voltage !== undefined) {
      resolved.BMS_Voltage = msg.BMS_Voltage;
    }
    else {
      resolved.BMS_Voltage = 0
    }

    if (msg.BMS_Current !== undefined) {
      resolved.BMS_Current = msg.BMS_Current;
    }
    else {
      resolved.BMS_Current = 0
    }

    if (msg.BMS_voltageDiff !== undefined) {
      resolved.BMS_voltageDiff = msg.BMS_voltageDiff;
    }
    else {
      resolved.BMS_voltageDiff = 0
    }

    if (msg.BMS_tempDiff !== undefined) {
      resolved.BMS_tempDiff = msg.BMS_tempDiff;
    }
    else {
      resolved.BMS_tempDiff = 0
    }

    if (msg.BMS_BCUMode !== undefined) {
      resolved.BMS_BCUMode = msg.BMS_BCUMode;
    }
    else {
      resolved.BMS_BCUMode = 0
    }

    if (msg.BMS_Alarm !== undefined) {
      resolved.BMS_Alarm = msg.BMS_Alarm;
    }
    else {
      resolved.BMS_Alarm = 0
    }

    if (msg.M_State !== undefined) {
      resolved.M_State = msg.M_State;
    }
    else {
      resolved.M_State = new Array(2).fill(0)
    }

    if (msg.M_Alarm !== undefined) {
      resolved.M_Alarm = msg.M_Alarm;
    }
    else {
      resolved.M_Alarm = new Array(2).fill(0)
    }

    if (msg.M_Power !== undefined) {
      resolved.M_Power = msg.M_Power;
    }
    else {
      resolved.M_Power = new Array(2).fill(0)
    }

    if (msg.M_Voltage !== undefined) {
      resolved.M_Voltage = msg.M_Voltage;
    }
    else {
      resolved.M_Voltage = new Array(2).fill(0)
    }

    if (msg.M_Current !== undefined) {
      resolved.M_Current = msg.M_Current;
    }
    else {
      resolved.M_Current = new Array(2).fill(0)
    }

    if (msg.M_Comm !== undefined) {
      resolved.M_Comm = msg.M_Comm;
    }
    else {
      resolved.M_Comm = new Array(2).fill(0)
    }

    if (msg.Relay_Temp !== undefined) {
      resolved.Relay_Temp = msg.Relay_Temp;
    }
    else {
      resolved.Relay_Temp = 0.0
    }

    if (msg.PowerConn_Temp !== undefined) {
      resolved.PowerConn_Temp = msg.PowerConn_Temp;
    }
    else {
      resolved.PowerConn_Temp = 0.0
    }

    if (msg.PreCharge_Temp !== undefined) {
      resolved.PreCharge_Temp = msg.PreCharge_Temp;
    }
    else {
      resolved.PreCharge_Temp = 0.0
    }

    if (msg.Power_Supply_Temp !== undefined) {
      resolved.Power_Supply_Temp = msg.Power_Supply_Temp;
    }
    else {
      resolved.Power_Supply_Temp = 0.0
    }

    if (msg.Battery_In_Volt !== undefined) {
      resolved.Battery_In_Volt = msg.Battery_In_Volt;
    }
    else {
      resolved.Battery_In_Volt = 0.0
    }

    if (msg.PRE_CHARGE_Volt !== undefined) {
      resolved.PRE_CHARGE_Volt = msg.PRE_CHARGE_Volt;
    }
    else {
      resolved.PRE_CHARGE_Volt = 0.0
    }

    if (msg.HW_ESTOP_Volt !== undefined) {
      resolved.HW_ESTOP_Volt = msg.HW_ESTOP_Volt;
    }
    else {
      resolved.HW_ESTOP_Volt = 0.0
    }

    if (msg.SW_ESTOP_Volt !== undefined) {
      resolved.SW_ESTOP_Volt = msg.SW_ESTOP_Volt;
    }
    else {
      resolved.SW_ESTOP_Volt = 0.0
    }

    if (msg.Motor_Alarm_Code !== undefined) {
      resolved.Motor_Alarm_Code = msg.Motor_Alarm_Code;
    }
    else {
      resolved.Motor_Alarm_Code = new Array(2).fill(0)
    }

    if (msg.Motor_Fun_Fail !== undefined) {
      resolved.Motor_Fun_Fail = msg.Motor_Fun_Fail;
    }
    else {
      resolved.Motor_Fun_Fail = new Array(2).fill(0)
    }

    if (msg.M_INIT !== undefined) {
      resolved.M_INIT = msg.M_INIT;
    }
    else {
      resolved.M_INIT = new Array(2).fill(0)
    }

    if (msg.M_CW !== undefined) {
      resolved.M_CW = msg.M_CW;
    }
    else {
      resolved.M_CW = new Array(2).fill(0)
    }

    if (msg.M_CCW !== undefined) {
      resolved.M_CCW = msg.M_CCW;
    }
    else {
      resolved.M_CCW = new Array(2).fill(0)
    }

    if (msg.M_STOP !== undefined) {
      resolved.M_STOP = msg.M_STOP;
    }
    else {
      resolved.M_STOP = new Array(2).fill(0)
    }

    if (msg.M_BREAK !== undefined) {
      resolved.M_BREAK = msg.M_BREAK;
    }
    else {
      resolved.M_BREAK = new Array(2).fill(0)
    }

    if (msg.M_SETRPM !== undefined) {
      resolved.M_SETRPM = msg.M_SETRPM;
    }
    else {
      resolved.M_SETRPM = new Array(2).fill(0)
    }

    if (msg.M_GETRPM !== undefined) {
      resolved.M_GETRPM = msg.M_GETRPM;
    }
    else {
      resolved.M_GETRPM = new Array(2).fill(0)
    }

    if (msg.M_PARAM !== undefined) {
      resolved.M_PARAM = msg.M_PARAM;
    }
    else {
      resolved.M_PARAM = new Array(2).fill(0)
    }

    if (msg.M_ALARM !== undefined) {
      resolved.M_ALARM = msg.M_ALARM;
    }
    else {
      resolved.M_ALARM = new Array(2).fill(0)
    }

    if (msg.PRECHARGE_FUSE !== undefined) {
      resolved.PRECHARGE_FUSE = msg.PRECHARGE_FUSE;
    }
    else {
      resolved.PRECHARGE_FUSE = 0
    }

    if (msg.MOTOR_FUSE !== undefined) {
      resolved.MOTOR_FUSE = msg.MOTOR_FUSE;
    }
    else {
      resolved.MOTOR_FUSE = 0
    }

    if (msg.LED_STATE !== undefined) {
      resolved.LED_STATE = msg.LED_STATE;
    }
    else {
      resolved.LED_STATE = 0
    }

    return resolved;
    }
};

module.exports = Diagnostics;

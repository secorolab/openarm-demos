#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

namespace openarm_compl_ctrl {

constexpr int NUM_JOINTS = 7;

struct ArmHardwareConfig {
    std::string can_interface;
    std::array<openarm::damiao_motor::MotorType, NUM_JOINTS> motor_types;
    std::array<uint32_t, NUM_JOINTS> send_can_ids;
    std::array<uint32_t, NUM_JOINTS> recv_can_ids;
    std::array<openarm::damiao_motor::ControlMode, NUM_JOINTS> control_modes;
    openarm::damiao_motor::MotorType gripper_motor_type;
    uint32_t gripper_send_can_id;
    uint32_t gripper_recv_can_id;
    openarm::damiao_motor::ControlMode gripper_control_mode;
};

struct ArmState {
    double q[NUM_JOINTS] = {};
    double qd[NUM_JOINTS] = {};
    double qdd[NUM_JOINTS] = {};
    double tau_mes[NUM_JOINTS] = {};

    std::vector<openarm::damiao_motor::MITParam> mit_cmd;
};

struct GripperState {
    double q = 0.0;
    double qd = 0.0;
    double tau_mes = 0.0;

    openarm::damiao_motor::MITParam mit_cmd = {};
};


// Initialize motors, gripper and query params
bool openarm_init(openarm::can::socket::OpenArm& arm,
                  const ArmHardwareConfig& config);

// Enable motors (set callback mode to STATE and enable)
bool openarm_start(openarm::can::socket::OpenArm& arm);

// Disable motors (best-effort multiple disables)
void openarm_shutdown(openarm::can::socket::OpenArm& arm);

// Send torque commands (from ArmState/GripperState) and read back joint states
void openarm_update(
    openarm::can::socket::OpenArm& arm,
    ArmState& arm_state,
    GripperState& gripper_state);

// Return a default hardware config for an arm. The returned config uses
// sensible default motor types, CAN IDs and control modes. The caller may
// override fields after calling this function (e.g. to change interface).
ArmHardwareConfig make_default_arm_config(const std::string& can_interface = "can1");

} // namespace openarm_compl_ctrl

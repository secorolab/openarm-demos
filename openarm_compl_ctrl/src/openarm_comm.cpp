#include "openarm_compl_ctrl/openarm_comm.hpp"
#include <iostream>
#include <thread>

namespace openarm_compl_ctrl {

bool openarm_init(openarm::can::socket::OpenArm& arm,
                  const ArmHardwareConfig& config)
{
    std::cout << "Configuring OpenArm CAN communication on interface " << config.can_interface << "..." << std::endl;

    std::vector<openarm::damiao_motor::MotorType> motor_types_vec(
        config.motor_types.begin(), config.motor_types.end());
    std::vector<uint32_t> send_can_ids_vec(
        config.send_can_ids.begin(), config.send_can_ids.end());
    std::vector<uint32_t> recv_can_ids_vec(
        config.recv_can_ids.begin(), config.recv_can_ids.end());
    std::vector<openarm::damiao_motor::ControlMode> control_modes_vec(
        config.control_modes.begin(), config.control_modes.end());

    std::cout << "Initializing arm's motors..." << std::endl;
    arm.init_arm_motors(motor_types_vec, send_can_ids_vec, recv_can_ids_vec, control_modes_vec);

    std::cout << "Initializing arm's gripper..." << std::endl;
    arm.init_gripper_motor(
        config.gripper_motor_type,
        config.gripper_send_can_id,
        config.gripper_recv_can_id,
        config.gripper_control_mode);

    std::cout << "Querying motor Recv IDs..." << std::endl;
    arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::PARAM);
    arm.query_param_all(static_cast<int>(openarm::damiao_motor::RID::MST_ID));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    arm.recv_all(2000);

    std::cout << config.can_interface << " motors:" << std::endl;
    for (const auto& motor : arm.get_arm().get_motors()) {
        std::cout << "  " << motor.get_send_can_id()
                  << " id: " << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
                  << " Queried Control Mode: " << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
                  << std::endl;
    }
    for (const auto& motor : arm.get_gripper().get_motors()) {
        std::cout << "  G: " << motor.get_send_can_id()
                  << " id: " << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
                  << " Queried Control Mode: " << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
                  << std::endl;
    }

    std::cout << "Arm initialization complete." << std::endl;
    return true;
}

ArmHardwareConfig make_default_arm_config(const std::string& can_interface) {
    ArmHardwareConfig cfg;
    cfg.can_interface = can_interface;
    cfg.motor_types = { {
        openarm::damiao_motor::MotorType::DM8009,
        openarm::damiao_motor::MotorType::DM8009,
        openarm::damiao_motor::MotorType::DM4340,
        openarm::damiao_motor::MotorType::DM4340,
        openarm::damiao_motor::MotorType::DM4310,
        openarm::damiao_motor::MotorType::DM4310,
        openarm::damiao_motor::MotorType::DM4310
    } };
    cfg.gripper_motor_type = openarm::damiao_motor::MotorType::DM4310;

    cfg.send_can_ids = { {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    } };
    cfg.recv_can_ids = { {
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
    } };

    cfg.gripper_send_can_id = 0x08;
    cfg.gripper_recv_can_id = 0x18;

    cfg.control_modes = { {
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT
    } };
    cfg.gripper_control_mode = openarm::damiao_motor::ControlMode::MIT;

    return cfg;
}

bool openarm_start(openarm::can::socket::OpenArm& arm) {
    std::cout << "Enabling arm's motors..." << std::endl;
    arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    arm.enable_all();
    arm.recv_all(2000);
    return true;
}

void openarm_shutdown(openarm::can::socket::OpenArm& arm) {
    std::cout << "Shutting down arm..." << std::endl;
    for (int attempt = 0; attempt < 3; ++attempt) {
        arm.disable_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        arm.recv_all(2000);
    }
}

void openarm_update(
    openarm::can::socket::OpenArm& arm,
    ArmState& arm_state,
    GripperState& gripper_state)
{
    // Send MIT commands for arm and gripper (controller-provided)
    if (!arm_state.mit_cmd.empty()) {
        arm.get_arm().mit_control_all(arm_state.mit_cmd);
    } else {
        // If no command provided, raise error
        std::cerr << "Warning: No MIT command provided for arm joints. Sending zero commands." << std::endl;
        arm.get_arm().mit_control_all(std::vector<openarm::damiao_motor::MITParam>(NUM_JOINTS, openarm::damiao_motor::MITParam{}));
    }

    arm.get_gripper().mit_control_all(std::vector<openarm::damiao_motor::MITParam>{gripper_state.mit_cmd});
    
    arm.recv_all(100);

    for (int i = 0; i < NUM_JOINTS; ++i) {
        arm_state.q[i] = arm.get_arm().get_motors()[i].get_position();
        arm_state.qd[i] = arm.get_arm().get_motors()[i].get_velocity();
        arm_state.tau_mes[i] = arm.get_arm().get_motors()[i].get_torque();
    }
    gripper_state.q = arm.get_gripper().get_motors()[0].get_position();
    gripper_state.qd = arm.get_gripper().get_motors()[0].get_velocity();
    gripper_state.tau_mes = arm.get_gripper().get_motors()[0].get_torque();
}

} // namespace openarm_compl_ctrl

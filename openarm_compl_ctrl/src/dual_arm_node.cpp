#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include <rclcpp/rclcpp.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <kdl_parser/kdl_parser.hpp>

#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/kinfam_io.hpp"
#include "kdl/frames_io.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl/chainidsolver_recursive_newton_euler.hpp"

#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>


#define LOG_INFO(node, msg, ...) RCLCPP_INFO(node->get_logger(), msg, ##__VA_ARGS__)
#define LOG_ERROR(node, msg, ...) RCLCPP_ERROR(node->get_logger(), msg, ##__VA_ARGS__)

#define LOG_INFO_S(node, expr) RCLCPP_INFO_STREAM((node)->get_logger(), expr)
#define LOG_ERROR_S(node, expr) RCLCPP_ERROR_STREAM((node)->get_logger(), expr)

#define NUM_JOINTS 7

std::atomic_bool shutting_down{false};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("openarm_compl_ctrl_dual_arm_node");

    rclcpp::on_shutdown([&]() {
        shutting_down.store(true);
    });

    std::string urdf_file = "openarm_bimanual.urdf";
    std::string urdf_path = ament_index_cpp::get_package_share_directory("openarm_compl_ctrl") + "/urdf/" + urdf_file;

    // -------------------- KDL Setup --------------------
    KDL::Tree kdl_tree;
    if (!kdl_parser::treeFromFile(urdf_path, kdl_tree)) {
        LOG_ERROR(node, "Failed to parse URDF file: %s", urdf_path.c_str());
        return -1;
    }

    KDL::Chain left_arm_chain, right_arm_chain;
    if (!kdl_tree.getChain("world", "openarm_left_hand_tcp", left_arm_chain) ||
        !kdl_tree.getChain("world", "openarm_right_hand_tcp", right_arm_chain)) {
        LOG_ERROR(node, "Failed to extract KDL chains from URDF");
        return -1;
    }

    LOG_INFO(node, "Successfully parsed URDF and extracted KDL chains");

    int num_joints_left = left_arm_chain.getNrOfJoints();
    int num_joints_right = right_arm_chain.getNrOfJoints();

    assert(num_joints_left == NUM_JOINTS && "Left arm has unexpected number of joints");
    assert(num_joints_right == NUM_JOINTS && "Right arm has unexpected number of joints");

    // print link names and joint names for debugging
    LOG_INFO(node, "Left arm has %d joints:", num_joints_left);
    for (size_t i = 0; i < left_arm_chain.getNrOfSegments(); ++i) {
        const auto& segment = left_arm_chain.getSegment(i);
        LOG_INFO(node, "  Segment %zu: %s", i, segment.getName().c_str());
        if (segment.getJoint().getType() != KDL::Joint::None) {
            LOG_INFO(node, "    Joint: %s", segment.getJoint().getName().c_str());
        }
    }
    LOG_INFO(node, "Right arm has %d joints:", num_joints_right);
    for (size_t i = 0; i < right_arm_chain.getNrOfSegments(); ++i) {
        const auto& segment = right_arm_chain.getSegment(i);
        LOG_INFO(node, "  Segment %zu: %s", i, segment.getName().c_str());
        if (segment.getJoint().getType() != KDL::Joint::None) {
            LOG_INFO(node, "    Joint: %s", segment.getJoint().getName().c_str());
        }
    }

    // -------------------- OpenArm CAN Params --------------------
    constexpr std::array<openarm::damiao_motor::MotorType, 7> MOTOR_TYPES = {
        openarm::damiao_motor::MotorType::DM8009,
        openarm::damiao_motor::MotorType::DM8009,
        openarm::damiao_motor::MotorType::DM4340,
        openarm::damiao_motor::MotorType::DM4340,
        openarm::damiao_motor::MotorType::DM4310,
        openarm::damiao_motor::MotorType::DM4310,
        openarm::damiao_motor::MotorType::DM4310
    };
    constexpr openarm::damiao_motor::MotorType GRIPPER_MOTOR_TYPE = openarm::damiao_motor::MotorType::DM4310;

    constexpr std::array<uint32_t, 7> SEND_CAN_IDS = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
    };

    constexpr std::array<uint32_t, 7> RECV_CAN_IDS = {
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
    };

    constexpr uint32_t GRIPPER_SEND_CAN_ID = 0x08;
    constexpr uint32_t GRIPPER_RECV_CAN_ID = 0x18;

    // -------------------- OpenArm CAN Setup --------------------
    std::vector<openarm::damiao_motor::MotorType> motor_types_vec(MOTOR_TYPES.begin(), MOTOR_TYPES.end());
    std::vector<uint32_t> send_can_ids_vec(SEND_CAN_IDS.begin(), SEND_CAN_IDS.end());
    std::vector<uint32_t> recv_can_ids_vec(RECV_CAN_IDS.begin(), RECV_CAN_IDS.end());
    std::vector<openarm::damiao_motor::ControlMode> control_modes = {
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT,
        openarm::damiao_motor::ControlMode::MIT
    };

    LOG_INFO(node, "Configuring OpenArm CAN communication...");
    openarm::can::socket::OpenArm right_arm("can0", true); // true - CAN-FD
    openarm::can::socket::OpenArm left_arm("can1", true);  // true - CAN-FD

    LOG_INFO(node, "Initializing arm's motors...");
    right_arm.init_arm_motors(motor_types_vec, send_can_ids_vec, recv_can_ids_vec, control_modes);
    left_arm.init_arm_motors(motor_types_vec, send_can_ids_vec, recv_can_ids_vec, control_modes);

    LOG_INFO(node, "Initializing arm's grippers...");
    right_arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_CAN_ID, GRIPPER_RECV_CAN_ID, openarm::damiao_motor::ControlMode::MIT);
    left_arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_CAN_ID, GRIPPER_RECV_CAN_ID, openarm::damiao_motor::ControlMode::MIT);

    // Set callback mode to ignore and enable all motors
    // right_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::IGNORE);
    // left_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::IGNORE);
    
    // Set device mode to param and query motor id
    LOG_INFO(node, "Querying motor Recv IDs...");
    right_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::PARAM);
    left_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::PARAM);
    right_arm.query_param_all(static_cast<int>(openarm::damiao_motor::RID::MST_ID));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    left_arm.query_param_all(static_cast<int>(openarm::damiao_motor::RID::MST_ID));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Wait 2ms for responses
    right_arm.recv_all(2000);
    left_arm.recv_all(2000);

    // Access motors through components
    LOG_INFO(node, "Right arm motors:");
    for (const auto& motor : right_arm.get_arm().get_motors()) {
        LOG_INFO_S(node,
            "  " << motor.get_send_can_id()
            << " id: "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
            << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
        );
    }
    for (const auto& motor : right_arm.get_gripper().get_motors()) {
        LOG_INFO_S(node,
            "  G: " << motor.get_send_can_id()
            << " id: "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
            << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
        );
    }
    LOG_INFO(node, "Left arm motors:");
    for (const auto& motor : left_arm.get_arm().get_motors()) {
        LOG_INFO_S(node,
            "  " << motor.get_send_can_id()
            << " id: "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
            << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
        );
    }
    for (const auto& motor : left_arm.get_gripper().get_motors()) {
        LOG_INFO_S(node,
            " G: " << motor.get_send_can_id()
            << " id: "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
            << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
            << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
        );
    }
    
    LOG_INFO(node, "Enabling arm's motors...");
    right_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    left_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    right_arm.enable_all();
    left_arm.enable_all();
    // Wait 2ms for motors to be fully enabled - enabling is slow op.
    right_arm.recv_all(2000);
    left_arm.recv_all(2000);

    // Set device mode to state - to control motors
    LOG_INFO(node, "Setting motors to state control mode...");

    right_arm.get_arm().mit_control_all({
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    });
    left_arm.get_arm().mit_control_all({
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    });

    right_arm.get_gripper().mit_control_all({
        openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    });
    
    float left_gripper_open = -1.3f;
    float left_gripper_close = -0.3f;
    float left_gripper_tau = 0.25f;
    constexpr float deadband = 0.03f;
    bool left_gripper_closing = false;

    {
        const float pos =
            left_arm.get_gripper().get_motors().front().get_position();
        const float mid = 0.5f * (left_gripper_open + left_gripper_close);
        left_gripper_closing = (pos > mid);
    }

    rclcpp::WallRate rate(100);
    while (rclcpp::ok() && !shutting_down.load()) {

        right_arm.refresh_all();
        right_arm.recv_all();
        left_arm.refresh_all();
        left_arm.recv_all();

        const float pos =
            left_arm.get_gripper().get_motors().front().get_position();

        if (left_gripper_closing) {
            if (pos >= left_gripper_close - deadband) {
                left_gripper_closing = false;
            }
        } else {
            if (pos <= left_gripper_open + deadband) {
                left_gripper_closing = true;
            }
        }

        const float tau =
            left_gripper_closing ? left_gripper_tau : -left_gripper_tau;

        left_arm.get_gripper().mit_control_all({
            openarm::damiao_motor::MITParam{0, 0, 0, 0, tau}
        });

        rate.sleep();
    }


    // shutdown
    LOG_INFO(node, "Disabling arm's motors...");
    right_arm.disable_all();
    left_arm.disable_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    right_arm.recv_all(2000);
    left_arm.recv_all(2000);

    LOG_INFO(node, "Shutting down node...");
    rclcpp::shutdown();
    return 0;
}

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
#include "kdl/treefksolverpos_recursive.hpp"
#include "kdl/chainfksolvervel_recursive.hpp"
#include "kdl/chainhdsolver_vereshchagin.hpp"
#include "kdl/chainhdsolver_vereshchagin_fext.hpp"

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
    KDL::Chain world_left_base_chain, world_right_base_chain;
    if (!kdl_tree.getChain("openarm_left_link0", "openarm_left_link7", left_arm_chain) ||
        !kdl_tree.getChain("openarm_right_link0", "openarm_right_link7", right_arm_chain ) ||
        !kdl_tree.getChain("world", "openarm_left_link0", world_left_base_chain) ||
        !kdl_tree.getChain("world", "openarm_right_link0", world_right_base_chain)) {
        LOG_ERROR(node, "Failed to extract KDL chains from URDF");
        return -1;
    }

    KDL::Vector gravity_vec(0, 0, -9.81);

    KDL::TreeFkSolverPos_recursive fk_solver(kdl_tree);
    KDL::JntArray q(kdl_tree.getNrOfJoints());
    
    KDL::Frame f_left_base, f_right_base;
    fk_solver.JntToCart(q, f_left_base, "openarm_left_link0");
    fk_solver.JntToCart(q, f_right_base, "openarm_right_link0");
 
    KDL::Vector g_left_base = f_left_base.M.Inverse() * gravity_vec;
    KDL::Vector g_right_base = f_right_base.M.Inverse() * gravity_vec;

    LOG_INFO(node, "Successfully parsed URDF and extracted KDL chains");

    constexpr double ee_mass = 0.42205099999999995;

    int num_joints_left = left_arm_chain.getNrOfJoints();
    int num_joints_right = right_arm_chain.getNrOfJoints();

    int num_segments_left = left_arm_chain.getNrOfSegments();
    int num_segments_right = right_arm_chain.getNrOfSegments();

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

    assert(num_joints_left == NUM_JOINTS && "Left arm has unexpected number of joints");
    assert(num_joints_right == NUM_JOINTS && "Right arm has unexpected number of joints");

    LOG_INFO(node, "Left arm chain has %d segments and %d joints", num_segments_left, num_joints_left);

    assert(num_segments_left == num_joints_left && "Left arm has unexpected number of segments");
    assert(num_segments_right == num_joints_right && "Right arm has unexpected number of segments");
    
    LOG_INFO_S(node, "Left arm gravity in base frame: " << g_left_base);
    LOG_INFO_S(node, "Right arm gravity in base frame: " << g_right_base);

#ifndef SKIP_ROBOT_COMM

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
    // openarm::can::socket::OpenArm right_arm("can0", true); // true - CAN-FD
    openarm::can::socket::OpenArm left_arm("can1", true);  // true - CAN-FD

    LOG_INFO(node, "Initializing arm's motors...");
    // right_arm.init_arm_motors(motor_types_vec, send_can_ids_vec, recv_can_ids_vec, control_modes);
    left_arm.init_arm_motors(motor_types_vec, send_can_ids_vec, recv_can_ids_vec, control_modes);

    LOG_INFO(node, "Initializing arm's grippers...");
    // right_arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_CAN_ID, GRIPPER_RECV_CAN_ID, openarm::damiao_motor::ControlMode::MIT);
    left_arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_CAN_ID, GRIPPER_RECV_CAN_ID, openarm::damiao_motor::ControlMode::MIT);

    // Set device mode to param and query motor id
    LOG_INFO(node, "Querying motor Recv IDs...");
    // right_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::PARAM);
    left_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::PARAM);
    // right_arm.query_param_all(static_cast<int>(openarm::damiao_motor::RID::MST_ID));
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    left_arm.query_param_all(static_cast<int>(openarm::damiao_motor::RID::MST_ID));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Wait 2ms for responses
    // right_arm.recv_all(2000);
    left_arm.recv_all(2000);

    // Access motors through components
    // LOG_INFO(node, "Right arm motors:");
    // for (const auto& motor : right_arm.get_arm().get_motors()) {
    //     LOG_INFO_S(node,
    //         "  " << motor.get_send_can_id()
    //         << " id: "
    //         << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
    //         << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
    //         << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
    //     );
    // }
    // for (const auto& motor : right_arm.get_gripper().get_motors()) {
    //     LOG_INFO_S(node,
    //         "  G: " << motor.get_send_can_id()
    //         << " id: "
    //         << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::MST_ID))
    //         << " Queried Control Mode (1: MIT, 2: POS_VEL, 3: VEL, 4: POS_FORCE): "
    //         << motor.get_param(static_cast<int>(openarm::damiao_motor::RID::CTRL_MODE))
    //     );
    // }
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
    // right_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    left_arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    // right_arm.enable_all();
    left_arm.enable_all();
    // Wait 2ms for motors to be fully enabled - enabling is slow op.
    // right_arm.recv_all(2000);
    left_arm.recv_all(2000);

#endif

    // --------------------- Vereshchagin Config ---------------------
    KDL::Twist g_left_arm(
        KDL::Vector(-g_left_base.x(), -g_left_base.y(), -g_left_base.z()),
        KDL::Vector::Zero()
    );
    KDL::Twist g_right_arm(
        KDL::Vector(-g_right_base.x(), -g_right_base.y(), -g_right_base.z()),
        KDL::Vector::Zero()
    );

    KDL::JntArray q_left(num_joints_left);
    KDL::JntArray qd_left(num_joints_left);
    KDL::JntArray qdd_left(num_joints_left);

    // KDL::JntArray q_right(num_joints_right);
    // KDL::JntArray qd_right(num_joints_right);
    // KDL::JntArray qdd_right(num_joints_right);

    int num_constraints = 6;

    KDL::ChainHdSolver_Vereshchagin achd_solver_left(left_arm_chain, g_left_arm, num_constraints);
    // KDL::ChainHdSolver_Vereshchagin achd_solver_right(right_arm_chain, g_right_arm, num_constraints);

    KDL::JntArray ff_taus(num_joints_left);

    KDL::Wrenches f_ext(num_segments_left);
    for (size_t i = 0; i < f_ext.size(); ++i) {
        f_ext[i] = KDL::Wrench::Zero();
    }

    KDL::Jacobian alpha_left_world(num_constraints);
    KDL::JntArray beta_left(num_constraints);

    // KDL::Jacobian alpha_right_world(num_constraints);
    // KDL::JntArray beta_right(num_constraints);

    alpha_left_world.setColumn(0, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0)));
    alpha_left_world.setColumn(1, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1))); 
           
    // alpha_right_world.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    // alpha_right_world.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0)));
    // alpha_right_world.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0)));
    // alpha_right_world.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0)));
    // alpha_right_world.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0)));
    // alpha_right_world.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));

    KDL::JntArray ctau_left(num_joints_left);
    // KDL::JntArray ctau_right(num_joints_right);

    // transform world to respective arm base frame
    KDL::Frame f_left_base_inv = f_left_base.Inverse();
    // KDL::Frame f_right_base_inv = f_right_base.Inverse();

    KDL::Jacobian alpha_left(num_constraints);
    // KDL::Jacobian alpha_right(num_constraints);
    for (int i = 0; i < num_constraints; ++i) {
        alpha_left.setColumn(i, f_left_base_inv * alpha_left_world.getColumn(i));
        // alpha_right.setColumn(i, f_right_base_inv * alpha_right_world.getColumn(i));
    }

    KDL::ChainFkSolverVel_recursive fk_vel_solver_left(left_arm_chain);
    // KDL::ChainFkSolverVel_recursive fk_vel_solver_right(right_arm_chain);

    KDL::FrameVel left_ee_vel, right_ee_vel;
    KDL::JntArrayVel left_q_qd(num_joints_left);
    // KDL::JntArrayVel right_q_qd(num_joints_right);

    fk_vel_solver_left.JntToCart(left_q_qd, left_ee_vel);
    // fk_vel_solver_right.JntToCart(right_q_qd, right_ee_vel);

    int r = achd_solver_left.CartToJnt(q_left, qd_left, qdd_left, alpha_left, beta_left, f_ext, ff_taus, ctau_left);
    if (r < 0) {
        LOG_ERROR(node, "Failed to compute feedforward torques for left arm: %d", r);
        return -1;
    }


#ifndef SKIP_ROBOT_COMM

    // right_arm.get_arm().mit_control_all({
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    // });
    // left_arm.get_arm().mit_control_all({
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0},
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    // });
    //
    // right_arm.get_gripper().mit_control_all({
    //     openarm::damiao_motor::MITParam{0, 0, 0, 0, 0}
    // });
    //
    // float left_gripper_open = -1.3f;
    // float left_gripper_close = -0.3f;
    // float left_gripper_tau = 0.25f;
    // constexpr float deadband = 0.03f;
    // bool left_gripper_closing = false;
    //
    // {
    //     const float pos =
    //         left_arm.get_gripper().get_motors().front().get_position();
    //     const float mid = 0.5f * (left_gripper_open + left_gripper_close);
    //     left_gripper_closing = (pos > mid);
    // }
    
    for (int i = 0; i < num_joints_left; ++i) {
        q_left(i) = left_arm.get_arm().get_motors()[i].get_position();
        qd_left(i) = left_arm.get_arm().get_motors()[i].get_velocity();
    }
    left_q_qd = KDL::JntArrayVel(q_left, qd_left);
    fk_vel_solver_left.JntToCart(left_q_qd, left_ee_vel);

    double left_ee_lin_vel_x_sp = 0.0; // m/s
    double left_ee_lin_vel_y_sp = 0.0; // m/s
    double left_ee_lin_vel_z_sp = 0.0; // m/s
    double left_ee_ang_vel_x_sp = 0.0; // rad/s
    double left_ee_ang_vel_y_sp = 0.0; // rad/s
    double left_ee_ang_vel_z_sp = 0.5; // rad/s

    LOG_INFO(node, "Starting control loop...");

    // 1khz
    auto desired_loop_rate = std::chrono::milliseconds(1);
    auto now = std::chrono::high_resolution_clock::now();
    auto deadline = now + desired_loop_rate;

    while (rclcpp::ok() && !shutting_down.load()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // right_arm.refresh_all();
        // right_arm.recv_all();
        left_arm.refresh_all();
        left_arm.recv_all();

        for (int i = 0; i < num_joints_left; ++i) {
            q_left(i) = left_arm.get_arm().get_motors()[i].get_position();
            qd_left(i) = left_arm.get_arm().get_motors()[i].get_velocity();
        }
        left_q_qd = KDL::JntArrayVel(q_left, qd_left);
        fk_vel_solver_left.JntToCart(left_q_qd, left_ee_vel);

        KDL::Vector left_ee_lin_vel = left_ee_vel.p.v;
        KDL::Vector left_ee_ang_vel = left_ee_vel.M.w;

        KDL::Vector left_ee_lin_vel_world = f_left_base.M * left_ee_lin_vel;
        KDL::Vector left_ee_ang_vel_world = f_left_base.M * left_ee_ang_vel;

        // std::cout << "ee twist (world): " << KDL::Twist(left_ee_lin_vel_world, left_ee_ang_vel_world) << std::endl;
        
        double left_ee_lin_vel_x = left_ee_lin_vel_world.x();
        double left_ee_lin_vel_y = left_ee_lin_vel_world.y();
        double left_ee_lin_vel_z = left_ee_lin_vel_world.z();
        double left_ee_ang_vel_x = left_ee_ang_vel_world.x();
        double left_ee_ang_vel_y = left_ee_ang_vel_world.y();
        double left_ee_ang_vel_z = left_ee_ang_vel_world.z();

        double left_ee_lin_vel_x_error = left_ee_lin_vel_x_sp - left_ee_lin_vel_x;
        double left_ee_lin_vel_y_error = left_ee_lin_vel_y_sp - left_ee_lin_vel_y;
        double left_ee_lin_vel_z_error = left_ee_lin_vel_z_sp - left_ee_lin_vel_z;
        double left_ee_ang_vel_x_error = left_ee_ang_vel_x_sp - left_ee_ang_vel_x;
        double left_ee_ang_vel_y_error = left_ee_ang_vel_y_sp - left_ee_ang_vel_y;
        double left_ee_ang_vel_z_error = left_ee_ang_vel_z_sp - left_ee_ang_vel_z;

        double left_ee_lin_vel_x_kp = 10.0;
        double left_ee_lin_vel_y_kp = 10.0;
        double left_ee_lin_vel_z_kp = 10.0;
        double left_ee_ang_vel_x_kp = 10.0;
        double left_ee_ang_vel_y_kp = 10.0;
        double left_ee_ang_vel_z_kp = 400.0;
        
        double left_ee_lin_vel_x_cntrl_sig = left_ee_lin_vel_x_kp * left_ee_lin_vel_x_error;
        double left_ee_lin_vel_y_cntrl_sig = left_ee_lin_vel_y_kp * left_ee_lin_vel_y_error;
        double left_ee_lin_vel_z_cntrl_sig = left_ee_lin_vel_z_kp * left_ee_lin_vel_z_error;
        double left_ee_ang_vel_x_cntrl_sig = left_ee_ang_vel_x_kp * left_ee_ang_vel_x_error;
        double left_ee_ang_vel_y_cntrl_sig = left_ee_ang_vel_y_kp * left_ee_ang_vel_y_error;
        double left_ee_ang_vel_z_cntrl_sig = left_ee_ang_vel_z_kp * left_ee_ang_vel_z_error;
        
        beta_left(0) = left_ee_lin_vel_x_cntrl_sig;
        beta_left(1) = left_ee_lin_vel_y_cntrl_sig;
        beta_left(2) = left_ee_lin_vel_z_cntrl_sig;
        beta_left(3) = left_ee_ang_vel_x_cntrl_sig;
        beta_left(4) = left_ee_ang_vel_y_cntrl_sig;
        beta_left(5) = left_ee_ang_vel_z_cntrl_sig;

        achd_solver_left.CartToJnt(q_left, qd_left, qdd_left, alpha_left, beta_left, f_ext, ff_taus, ctau_left);
        // std::cout << "beta: " << beta_left << std::endl;
        // std::cout << "ctau: " << ctau_left << std::endl;
        // std::cout << std::endl;

        // update robot MIT control command
        left_arm.get_arm().mit_control_all({
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(0)},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(1)},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(2)},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(3)},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, -0.25},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(5)},
            openarm::damiao_motor::MITParam{0, 0, 0, 0, ctau_left(6)}
        });

        // rate.sleep();
        //
        //auto end_time = std::chrono::high_resolution_clock::now();
        //auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        while (now < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            now = std::chrono::high_resolution_clock::now();
        }
        while (deadline < now) {
            deadline += desired_loop_rate;
        }
        
        //while (loop_duration < desired_loop_rate) {
        //    std::this_thread::sleep_for(std::chrono::microseconds(100));
        //    end_time = std::chrono::high_resolution_clock::now();
        //    loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        //}

    }


    // shutdown
    LOG_INFO(node, "Disabling arm's motors...");
    // right_arm.disable_all();
    left_arm.disable_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // right_arm.recv_all(2000);
    left_arm.recv_all(2000);

#endif

    LOG_INFO(node, "Shutting down node...");
    rclcpp::shutdown();
    return 0;
}

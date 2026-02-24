#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <csignal>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "openarm_compl_ctrl/msg/pid_debug.hpp"

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
#include "kdl/chainhdsolver_vereshchagin_fixed_joint.hpp"
#include "kdl/chainhdsolver_vereshchagin_fext.hpp"

#include "openarm_compl_ctrl/openarm_comm.hpp"
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>


#define LOG_INFO(node, msg, ...) RCLCPP_INFO(node->get_logger(), msg, ##__VA_ARGS__)
#define LOG_ERROR(node, msg, ...) RCLCPP_ERROR(node->get_logger(), msg, ##__VA_ARGS__)

#define LOG_INFO_S(node, expr) RCLCPP_INFO_STREAM((node)->get_logger(), expr)
#define LOG_ERROR_S(node, expr) RCLCPP_ERROR_STREAM((node)->get_logger(), expr)

#define ASSERT(cond, node, msg, ...) \
    do { \
        if (!(cond)) { \
            RCLCPP_FATAL( \
                (node)->get_logger(), \
                "[%s:%d:%s] " msg, \
                __FILE__, __LINE__, __FUNCTION__, \
                ##__VA_ARGS__); \
            std::abort(); \
        } \
    } while (0)
            

#define NUM_JOINTS 7

std::atomic_bool shutting_down{false};

void signal_handler(int /*signum*/) {
    shutting_down.store(true);
}

void kdl_tau_to_mit(const KDL::JntArray& tau, std::vector<openarm::damiao_motor::MITParam>& mit_cmds) {
    for (size_t i = 0; i < tau.rows(); ++i) {
        mit_cmds[i] = openarm::damiao_motor::MITParam{0.0, 0.0, 0.0, 0.0, tau(i)};
    }
}

class LowPassFilter {
private:
    double alpha_;  // Smoothing factor (0 < alpha <= 1)
    double filtered_value_;
    bool initialized_;

public:
    LowPassFilter(double cutoff_freq, double sample_rate) 
        : filtered_value_(0.0), initialized_(false) {
        // Calculate alpha from cutoff frequency
        double rc = 1.0 / (2.0 * M_PI * cutoff_freq);
        double dt = 1.0 / sample_rate;
        alpha_ = dt / (rc + dt);
    }

    double update(double raw_value) {
        if (!initialized_) {
            filtered_value_ = raw_value;
            initialized_ = true;
        } else {
            filtered_value_ = alpha_ * raw_value + (1.0 - alpha_) * filtered_value_;
        }
        return filtered_value_;
    }

    void reset() {
        filtered_value_ = 0.0;
        initialized_ = false;
    }
};

double evaluate_equality_constraint(double quantity, double reference) {
    return reference - quantity;
}

double evaluate_less_than_constraint(double quantity, double threshold) {
    return (quantity <= threshold) ? 0.0 : quantity - threshold;
}

double evaluate_greater_than_constraint(double quantity, double threshold) {
    return (quantity >= threshold) ? 0.0 : threshold - quantity;
}

double evaluate_bilateral_constraint(double quantity, double lower, double upper) {
    if (quantity < lower)
        return lower - quantity;
    else if (quantity > upper)
        return quantity - upper;
    else
        return 0.0;
}


class PIDController {
public:
    PIDController(double kp, double ki, double kd,
                  double output_min = -1000.0, double output_max = 1000.0,
                  double derivative_lpf_cutoff = 10.0,
                  double sample_rate = 1000.0,
                  double integral_decay_rate = 0.9,
                  double friction_comp = 0.0,
                  double integrator_min = -1e9,
                  double integrator_max =  1e9)
        : kp_(kp), ki_(ki), kd_(kd),
          output_min_(output_min), output_max_(output_max),
          decay_rate_(integral_decay_rate),
          friction_comp_(friction_comp),
          integrator_min_(integrator_min),
          integrator_max_(integrator_max),
          integral_(0.0),
          previous_error_(0.0),
          first_update_(true),
          derivative_filter_(derivative_lpf_cutoff, sample_rate)
    {}

    double control(double error, double dt)
    {
        if (first_update_) {
            previous_error_ = error;
            first_update_ = false;
        }

        // ----- Proportional -----
        p_term_ = kp_ * error;

        // ----- Derivative (filtered) -----
        double raw_derivative = (error - previous_error_) / dt;
        d_term_ = kd_ * raw_derivative;
        // d_term_ = kd_ * derivative_filter_.update(raw_derivative);

        // ----- Integral Update (before computing output) -----
        if (ki_ != 0.0) {
            integral_ += error * dt;

            // Exponential decay when error near zero
            if (std::abs(error) < 1e-6) {
                integral_ *= std::exp(-decay_rate_ * dt);
            }

            integral_ = std::clamp(integral_, integrator_min_, integrator_max_);
        }

        i_term_ = ki_ * integral_;

        // ----- Static friction compensation -----
        double friction = 0.0;
        if (friction_comp_ > 0.0 && std::abs(error) > 1e-6) {
            friction = (error > 0.0 ? 1.0 : -1.0) * friction_comp_;
        }

        // ----- Output -----
        double output = p_term_ + i_term_ + d_term_ + friction;
        last_output_ = std::clamp(output, output_min_, output_max_);

        // ----- Anti-windup (back-calculation style) -----
        if (ki_ != 0.0 && output != last_output_) {
            integral_ = (last_output_ - p_term_ - d_term_ - friction) / ki_;
        }

        previous_error_ = error;
        last_error_ = error;

        return last_output_;
    }

    void get_debug_values(double& p, double& i, double& d, double& error, double& control_sig) const {
        p = p_term_;
        i = i_term_;
        d = d_term_;
        error = last_error_;
        control_sig = last_output_;
    }

private:
    double kp_, ki_, kd_;
    double output_min_, output_max_;
    double decay_rate_;
    double friction_comp_;
    double integrator_min_, integrator_max_;

    double integral_;
    double previous_error_;
    bool first_update_;

    double p_term_ = 0.0;
    double i_term_ = 0.0;
    double d_term_ = 0.0;
    double last_error_ = 0.0;
    double last_output_ = 0.0;

    LowPassFilter derivative_filter_;
};

struct PIDDebugData {
    double p = 0.0;
    double i = 0.0;
    double d = 0.0;
    double error = 0.0;
    double control_sig = 0.0;
};

using openarm_compl_ctrl::ArmState;
using openarm_compl_ctrl::GripperState;

struct State {
    ArmState left;
    GripperState left_gripper;
    ArmState right;
    GripperState right_gripper;

    KDL::JntArray left_tau_cmd{NUM_JOINTS};
    KDL::JntArray right_tau_cmd{NUM_JOINTS};

    KDL::FrameVel left_ee_fvel;
    KDL::FrameVel right_ee_fvel;

    double left_gripper_tau_cmd = 0.0;
    double right_gripper_tau_cmd = 0.0;

    PIDDebugData left_ee_ang_vel_z_pid_debug;

    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number = 0;
};

class RobotState {
private:
    mutable std::mutex state_mutex_;
    std::deque<State> state_queue_;
    const size_t MAX_QUEUE_SIZE = 100;  // ~100ms buffer at 1kHz
    uint64_t sequence_counter_ = 0;
    
public:
    void update(const State& new_state) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        State timestamped_state = new_state;
        timestamped_state.sequence_number = sequence_counter_++;
        timestamped_state.timestamp = std::chrono::high_resolution_clock::now();
        
        state_queue_.push_back(timestamped_state);
        
        // Prevent unbounded growth
        if (state_queue_.size() > MAX_QUEUE_SIZE) {
            state_queue_.pop_front();
        }
    }
    
    // Pop one state for publishing
    bool popState(State& state) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_queue_.empty()) {
            return false;
        }
        state = state_queue_.front();
        state_queue_.pop_front();
        return true;
    }
    
    // Pop multiple states (batch publishing)
    std::vector<State> popStates(size_t max_count) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        std::vector<State> states;
        size_t count = std::min(max_count, state_queue_.size());
        
        for (size_t i = 0; i < count; ++i) {
            states.push_back(state_queue_.front());
            state_queue_.pop_front();
        }
        
        return states;
    }
    
    // Get queue size for diagnostics
    size_t queueSize() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_queue_.size();
    }
    
    // Peek at latest without removing
    bool peekLatest(State& state) const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_queue_.empty()) {
            return false;
        }
        state = state_queue_.back();
        return true;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_queue_.clear();
    }
};


class OpenArmROSNode : public rclcpp::Node {
public:
    OpenArmROSNode(std::shared_ptr<RobotState> state) 
        : Node("openarm_node"), state_(state) {
        
        // Publishers
        joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            "/debug_js", 10);

        pid_debug_pub_ = create_publisher<openarm_compl_ctrl::msg::PIDDebug>(
            "/pid_debug", 10);

        // Publishing timer - can handle bursts
        publish_timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_hz_)),
            std::bind(&OpenArmROSNode::publishCallback, this));
    }

private:
    void publishCallback() {
        // Drain the queue and publish only the most recent state.
        // The control loop pushes at 1kHz but we publish at ~500Hz,
        // so we must skip stale entries to avoid an ever-growing backlog.
        State state;
        bool have_state = false;
        while (state_->popState(state)) {
            have_state = true;
        }
        if (have_state) {
            publishState(state);
        }
    }

    void publishState(const State& state) {
        sensor_msgs::msg::JointState msg;
        openarm_compl_ctrl::msg::PIDDebug pid_debug_msg;
        
        // Convert timestamp to ROS time
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            state.timestamp.time_since_epoch()).count();
        msg.header.stamp = rclcpp::Time(ns);
        msg.header.frame_id = "openarm_left_link0";
        
        msg.name = {"joint1", "joint2", "joint3", "joint4", 
                    "joint5", "joint6", "joint7"};
        msg.position.resize(7);
        msg.velocity.resize(7);
        msg.effort.resize(7);
        
        for (int i = 0; i < 7; ++i) {
            msg.position[i] = state.left.q[i];
            msg.velocity[i] = state.left.qd[i];
            msg.effort[i] = state.left_tau_cmd(i);
        }

        
        joint_state_pub_->publish(msg);
        // pid_debug_pub_->publish(pid_debug_msg);

        // Track publishing stats
        publish_count_++;
        last_published_seq_ = state.sequence_number;
    }

    std::shared_ptr<RobotState> state_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<openarm_compl_ctrl::msg::PIDDebug>::SharedPtr pid_debug_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    int publish_rate_hz_ = 100;

    uint64_t publish_count_ = 0;
    uint64_t last_published_seq_ = 0;
};


using openarm_compl_ctrl::ArmHardwareConfig;

int main(int argc, char **argv) {
    rclcpp::InitOptions init_options;
    init_options.shutdown_on_signal = false;
    rclcpp::init(argc, argv, init_options);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto robot_state = std::make_shared<RobotState>();
    auto node = std::make_shared<OpenArmROSNode>(robot_state);

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);

    std::thread ros_thread([&executor]() {
        executor.spin();
    });
    
    std::string urdf_file = "openarm_bimanual.urdf";
    std::string urdf_path = ament_index_cpp::get_package_share_directory("openarm_compl_ctrl") + "/urdf/" + urdf_file;

    // -------------------- KDL Setup --------------------
    KDL::Tree kdl_tree;
    if (!kdl_parser::treeFromFile(urdf_path, kdl_tree)) {
        LOG_ERROR(node, "Failed to parse URDF file: %s", urdf_path.c_str());
        return -1;
    }

    KDL::Chain leftarm_chain, rightarm_chain;

    constexpr const char* WORLD_LINK = "world";
    constexpr const char* leftarm_link7 = "openarm_left_link7";
    constexpr const char* leftarm_TCP_LINK = "openarm_left_hand_tcp";
    constexpr const char* rightarm_TCP_LINK = "openarm_right_hand_tcp";

    if (!kdl_tree.getChain(WORLD_LINK, leftarm_TCP_LINK, leftarm_chain) ||
        !kdl_tree.getChain(WORLD_LINK, rightarm_TCP_LINK, rightarm_chain)) {
        LOG_ERROR(node, "Failed to extract KDL chains from URDF");
        return -1;
    }
    LOG_INFO(node, "Successfully parsed URDF and extracted KDL chains");

    int nj_left = leftarm_chain.getNrOfJoints();
    int nj_right = rightarm_chain.getNrOfJoints();

    int ns_left = leftarm_chain.getNrOfSegments();
    int ns_right = rightarm_chain.getNrOfSegments();

    // print link names and joint names for debugging
    LOG_INFO(node, "Left arm has %d joints:", nj_left);
    for (size_t i = 0; i < leftarm_chain.getNrOfSegments(); ++i) {
        const auto& segment = leftarm_chain.getSegment(i);
        LOG_INFO(node, "  Segment %zu: %s", i, segment.getName().c_str());
        if (segment.getJoint().getType() != KDL::Joint::None) {
            LOG_INFO(node, "    Joint: %s", segment.getJoint().getName().c_str());
        }
    }
    LOG_INFO(node, "Right arm has %d joints:", nj_right);
    for (size_t i = 0; i < rightarm_chain.getNrOfSegments(); ++i) {
        const auto& segment = rightarm_chain.getSegment(i);
        LOG_INFO(node, "  Segment %zu: %s", i, segment.getName().c_str());
        if (segment.getJoint().getType() != KDL::Joint::None) {
            LOG_INFO(node, "    Joint: %s", segment.getJoint().getName().c_str());
        }
    }

    ASSERT(nj_left == NUM_JOINTS, node, "Left arm has unexpected number of joints");
    ASSERT(nj_right == NUM_JOINTS, node, "Right arm has unexpected number of joints");

    LOG_INFO(node, "Left arm chain has %d segments and %d joints", ns_left, nj_left);

    // ASSERT(ns_left == ns_right, node, "Left arm has unexpected number of segments");
    
    // --------------------- State ----------------------

    State state;
    const KDL::Vector GRAVITY_VEC(0, 0, -9.81);

    // --------------------- Vereshchagin Config ---------------------
    KDL::Twist gravity_twist(
        KDL::Vector(-GRAVITY_VEC.x(), -GRAVITY_VEC.y(), -GRAVITY_VEC.z()),
        KDL::Vector::Zero()
    );
    
    int num_constraints = 6;

    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd_solver_leftarm(leftarm_chain, gravity_twist, num_constraints);
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd_solver_rightarm(rightarm_chain, gravity_twist, num_constraints);

    KDL::JntArray q_leftarm(NUM_JOINTS);
    KDL::JntArray qd_leftarm(NUM_JOINTS);
    KDL::JntArray ff_tau_leftarm(NUM_JOINTS);
    KDL::JntArray qdd_leftarm(NUM_JOINTS);
    KDL::JntArray tau_cmd_leftarm(NUM_JOINTS);

    KDL::JntArray q_rightarm(NUM_JOINTS);
    KDL::JntArray qd_rightarm(NUM_JOINTS);
    KDL::JntArray ff_tau_rightarm(NUM_JOINTS);
    KDL::JntArray qdd_rightarm(NUM_JOINTS);
    KDL::JntArray tau_cmd_rightarm(NUM_JOINTS);

    KDL::Wrenches f_ext_leftarm(ns_left);
    KDL::Wrenches f_ext_rightarm(ns_right);

    for (int i = 0; i < ns_left; ++i) {
        f_ext_leftarm[i] = KDL::Wrench::Zero();
        f_ext_rightarm[i] = KDL::Wrench::Zero();
    }

    KDL::Jacobian alpha_leftarm(num_constraints);
    KDL::JntArray beta_leftarm(num_constraints);

    KDL::Jacobian alpha_rightarm(num_constraints);
    KDL::JntArray beta_rightarm(num_constraints);
    
    alpha_leftarm.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    alpha_leftarm.setColumn(1, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 0))); 
    alpha_leftarm.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0))); 
    alpha_leftarm.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0))); 
    alpha_leftarm.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0))); 
    alpha_leftarm.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1))); 
 
    alpha_rightarm.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    alpha_rightarm.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0))); 
    alpha_rightarm.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0))); 
    alpha_rightarm.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0))); 
    alpha_rightarm.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0))); 
    alpha_rightarm.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));

    int solver_res;

    solver_res = achd_solver_leftarm.CartToJnt(
                                    q_leftarm,
                                    qd_leftarm,
                                    qdd_leftarm,
                                    alpha_leftarm, beta_leftarm, 
                                    f_ext_leftarm, ff_tau_leftarm,
                                    tau_cmd_leftarm);
    if (solver_res < 0) {
        LOG_ERROR(node, "Failed to compute feedforward torques for left arm: %d", solver_res);
        return -1;
    }

    solver_res = achd_solver_rightarm.CartToJnt(
                                    q_rightarm,
                                    qd_rightarm,
                                    qdd_rightarm,
                                    alpha_rightarm, beta_rightarm, 
                                    f_ext_rightarm, ff_tau_rightarm,
                                    tau_cmd_rightarm);
    if (solver_res < 0) {
        LOG_ERROR(node, "Failed to compute feedforward torques for right arm: %d", solver_res);
        return -1;
    }

    KDL::ChainFkSolverVel_recursive fk_vel_solver_leftarm(leftarm_chain);
    KDL::ChainFkSolverVel_recursive fk_vel_solver_rightarm(rightarm_chain);

    // --------------------- Hardware Interface ---------------------

    ArmHardwareConfig rightarm_config = openarm_compl_ctrl::make_default_arm_config("can0");
    ArmHardwareConfig leftarm_config = openarm_compl_ctrl::make_default_arm_config("can1");

    openarm::can::socket::OpenArm leftarm(leftarm_config.can_interface, true);
    openarm::can::socket::OpenArm rightarm(rightarm_config.can_interface, true);
    
    if (!openarm_compl_ctrl::openarm_init(leftarm, leftarm_config)) {
        LOG_ERROR(node, "Failed to initialize left arm");
        return -1;
    }

    if (!openarm_compl_ctrl::openarm_init(rightarm, rightarm_config)) {
        LOG_ERROR(node, "Failed to initialize right arm");
        return -1;
    }

    if (!openarm_compl_ctrl::openarm_start(leftarm)) {
        LOG_ERROR(node, "Failed to start left arm");
        return -1;
    }

    if (!openarm_compl_ctrl::openarm_start(rightarm)) {
        LOG_ERROR(node, "Failed to start right arm");
        return -1;
    }

    // Left arm hardware state
    openarm_compl_ctrl::ArmState leftarm_state = {};
    leftarm_state.mit_cmd.resize(NUM_JOINTS);
    openarm_compl_ctrl::GripperState gleftarm_state = {};

    // Right arm hardware state
    openarm_compl_ctrl::ArmState rightarm_state = {};
    rightarm_state.mit_cmd.resize(NUM_JOINTS);
    openarm_compl_ctrl::GripperState grightarm_state = {};

    // Bootstrap: send zero torque to read initial joint positions
    KDL::JntArray zero_tau(NUM_JOINTS);
    kdl_tau_to_mit(zero_tau, leftarm_state.mit_cmd);
    gleftarm_state.mit_cmd = openarm::damiao_motor::MITParam{0.0, 0.0, 0.0, 0.0, 0.0};
    openarm_compl_ctrl::openarm_update(leftarm, leftarm_state, gleftarm_state);

    kdl_tau_to_mit(zero_tau, rightarm_state.mit_cmd);
    grightarm_state.mit_cmd = openarm::damiao_motor::MITParam{0.0, 0.0, 0.0, 0.0, 0.0};
    openarm_compl_ctrl::openarm_update(rightarm, rightarm_state, grightarm_state);

    // Copy initial sensor readings into shared state
    for (int i = 0; i < NUM_JOINTS; ++i) {
        state.left.q[i]      = leftarm_state.q[i];
        state.left.qd[i]     = leftarm_state.qd[i];
        state.left.tau_mes[i] = leftarm_state.tau_mes[i];

        state.right.q[i]      = rightarm_state.q[i];
        state.right.qd[i]     = rightarm_state.qd[i];
        state.right.tau_mes[i] = rightarm_state.tau_mes[i];
    }
    state.left_gripper.q       = gleftarm_state.q;
    state.left_gripper.qd      = gleftarm_state.qd;
    state.left_gripper.tau_mes = gleftarm_state.tau_mes;

    state.right_gripper.q       = grightarm_state.q;
    state.right_gripper.qd      = grightarm_state.qd;
    state.right_gripper.tau_mes = grightarm_state.tau_mes;
    
    // Left arm - linear axes
    PIDController left_ee_acc_x_pid(5.0, 3.0, 0.1);
    PIDController left_ee_acc_y_pid(10.0, 3.0, 0.1);
    PIDController left_ee_acc_z_pid(10.0, 3.0, 0.1);
    // Left arm - angular axes
    PIDController left_ee_ang_acc_x_pid(100.0, 30.0, 0.1);
    PIDController left_ee_ang_acc_y_pid(100.0, 30.0, 0.1);
    PIDController left_ee_ang_acc_z_pid(100.0, 30.0, 0.1);

    // Right arm - linear axes
    PIDController right_ee_acc_x_pid(10.0, 5.0, 0.1);
    PIDController right_ee_acc_y_pid(10.0, 5.0, 0.1);
    PIDController right_ee_acc_z_pid(10.0, 5.0, 0.1);
    // Right arm - angular axes
    PIDController right_ee_ang_acc_x_pid(10.0, 5.0, 0.1);
    PIDController right_ee_ang_acc_y_pid(10.0, 5.0, 0.1);
    PIDController right_ee_ang_acc_z_pid(10.0, 5.0, 0.1);

    // --------------------- Compute Initial EE Frames ---------------------

    // Seed joint arrays with the first sensor readings before running FK
    for (int i = 0; i < NUM_JOINTS; ++i) {
        q_leftarm(i)  = state.left.q[i];
        qd_leftarm(i) = state.left.qd[i];
        q_rightarm(i)  = state.right.q[i];
        qd_rightarm(i) = state.right.qd[i];
    }

    KDL::JntArrayVel q_qd_leftarm(q_leftarm, qd_leftarm);
    KDL::FrameVel    ee_fvel_leftarm;
    fk_vel_solver_leftarm.JntToCart(q_qd_leftarm, ee_fvel_leftarm);

    KDL::JntArrayVel q_qd_rightarm(q_rightarm, qd_rightarm);
    KDL::FrameVel    ee_fvel_rightarm;
    fk_vel_solver_rightarm.JntToCart(q_qd_rightarm, ee_fvel_rightarm);

    KDL::Frame ee_frame_leftarm_start  = ee_fvel_leftarm.GetFrame();
    KDL::Frame ee_frame_rightarm_start = ee_fvel_rightarm.GetFrame();

    KDL::Frame ee_frame_leftarm_desired  = ee_frame_leftarm_start;
    ee_frame_leftarm_desired.p += KDL::Vector(0.2, 0.0, 0.1);
    // rotate ee 90 degrees about world Z axis
    KDL::Rotation rot_desired = KDL::Rotation::RotZ(M_PI / 2.0);
    // ee_frame_leftarm_desired.M = rot_desired * ee_frame_leftarm_desired.M;
    
    KDL::Frame ee_frame_rightarm_desired = ee_frame_rightarm_start;
    ee_frame_rightarm_desired.p += KDL::Vector(0.0, 0.0, 0.1); // 10 cm up in Z

    double r, p, y;
    ee_frame_leftarm_start.M.GetRPY(r, p, y);
    std::cout << "[LEFT]  Current EE Pose: " << ee_frame_leftarm_start.p
              << " RPY(" << r << ", " << p << ", " << y << ")" << std::endl;
    ee_frame_leftarm_desired.M.GetRPY(r, p, y);
    std::cout << "[LEFT]  Desired EE Pose: " << ee_frame_leftarm_desired.p
              << " RPY(" << r << ", " << p << ", " << y << ")" << std::endl;

    ee_frame_rightarm_start.M.GetRPY(r, p, y);
    std::cout << "[RIGHT] Current EE Pose: " << ee_frame_rightarm_start.p
              << " RPY(" << r << ", " << p << ", " << y << ")" << std::endl;
    ee_frame_rightarm_desired.M.GetRPY(r, p, y);
    std::cout << "[RIGHT] Desired EE Pose: " << ee_frame_rightarm_desired.p
              << " RPY(" << r << ", " << p << ", " << y << ")" << std::endl;
    
    constexpr double TAU_MAX = 10.0;
    constexpr double DT      = 0.001; // 1000 Hz control loop

    LOG_INFO(node, "Starting control loop...");

    // DT-based loop timing
    auto desired_loop_rate = std::chrono::microseconds(static_cast<int>(DT * 1e6));
    auto now = std::chrono::high_resolution_clock::now();
    auto deadline = now + desired_loop_rate;

    while (!shutting_down.load()) {

        for (int i = 0; i < NUM_JOINTS; ++i) {
            q_leftarm(i)  = state.left.q[i];
            qd_leftarm(i) = state.left.qd[i];

            q_rightarm(i)  = state.right.q[i];
            qd_rightarm(i) = state.right.qd[i];
        }

        // ===== Forward Kinematics =====
        q_qd_leftarm = KDL::JntArrayVel(q_leftarm, qd_leftarm);
        fk_vel_solver_leftarm.JntToCart(q_qd_leftarm, ee_fvel_leftarm);

        q_qd_rightarm = KDL::JntArrayVel(q_rightarm, qd_rightarm);
        fk_vel_solver_rightarm.JntToCart(q_qd_rightarm, ee_fvel_rightarm);

        KDL::Frame ee_frame_leftarm  = ee_fvel_leftarm.GetFrame();
        KDL::Frame ee_frame_rightarm = ee_fvel_rightarm.GetFrame();

        KDL::Twist ee_pose_error_leftarm  = KDL::diff(ee_frame_leftarm_desired, ee_frame_leftarm);
        KDL::Twist ee_pose_error_rightarm = KDL::diff(ee_frame_rightarm_desired, ee_frame_rightarm);

        // -- Left arm --
        beta_leftarm(0) = left_ee_acc_x_pid.control(ee_pose_error_leftarm.vel.x(), DT);
        beta_leftarm(1) = left_ee_acc_y_pid.control(ee_pose_error_leftarm.vel.y(), DT);
        beta_leftarm(2) = left_ee_acc_z_pid.control(ee_pose_error_leftarm.vel.z(), DT);
        beta_leftarm(3) = left_ee_ang_acc_x_pid.control(ee_pose_error_leftarm.rot.x(), DT);
        beta_leftarm(4) = left_ee_ang_acc_y_pid.control(ee_pose_error_leftarm.rot.y(), DT);
        beta_leftarm(5) = left_ee_ang_acc_z_pid.control(ee_pose_error_leftarm.rot.z(), DT);

        // -- Right arm --
        beta_rightarm(0) = right_ee_acc_x_pid.control(ee_pose_error_rightarm.vel.x(), DT);
        beta_rightarm(1) = right_ee_acc_y_pid.control(ee_pose_error_rightarm.vel.y(), DT);
        beta_rightarm(2) = right_ee_acc_z_pid.control(ee_pose_error_rightarm.vel.z(), DT);
        beta_rightarm(3) = right_ee_ang_acc_x_pid.control(ee_pose_error_rightarm.rot.x(), DT);
        beta_rightarm(4) = right_ee_ang_acc_y_pid.control(ee_pose_error_rightarm.rot.y(), DT);
        beta_rightarm(5) = right_ee_ang_acc_z_pid.control(ee_pose_error_rightarm.rot.z(), DT);

       // ===== Vereshchagin Inverse Dynamics =====
        solver_res = achd_solver_leftarm.CartToJnt(
                                    q_leftarm,
                                    qd_leftarm,
                                    qdd_leftarm,
                                    alpha_leftarm, beta_leftarm,
                                    f_ext_leftarm, ff_tau_leftarm,
                                    tau_cmd_leftarm);
        if (solver_res < 0) {
            LOG_ERROR(node, "Vereshchagin CartToJnt failed for left arm: %d", solver_res);
        }

        solver_res = achd_solver_rightarm.CartToJnt(
                                    q_rightarm,
                                    qd_rightarm,
                                    qdd_rightarm,
                                    alpha_rightarm, beta_rightarm,
                                    f_ext_rightarm, ff_tau_rightarm,
                                    tau_cmd_rightarm);
        if (solver_res < 0) {
            LOG_ERROR(node, "Vereshchagin CartToJnt failed for right arm: %d", solver_res);
        }

        // ===== Clamp Torques =====
        for (int i = 0; i < NUM_JOINTS; ++i) {
            tau_cmd_leftarm(i)  = std::clamp(tau_cmd_leftarm(i),  -TAU_MAX, TAU_MAX);
            tau_cmd_rightarm(i) = std::clamp(tau_cmd_rightarm(i), -TAU_MAX, TAU_MAX);
        }

        // ===== Update Shared State =====
        state.left_tau_cmd  = tau_cmd_leftarm;
        state.right_tau_cmd = tau_cmd_rightarm;
        state.left_ee_fvel  = ee_fvel_leftarm;
        state.right_ee_fvel = ee_fvel_rightarm;
        robot_state->update(state);

        double r, p, y;
        ee_frame_leftarm.M.GetRPY(r, p, y);
        std::cout << "Left EE Pos:  " << ee_frame_leftarm.p << " RPY(" << r << ", " << p << ", " << y << ")" << std::endl;
        std::cout << "Left EE Err:  " << ee_pose_error_leftarm.vel << " / " << ee_pose_error_leftarm.rot << std::endl;
        std::cout << "Left EE Ctrl: " << beta_leftarm << std::endl;
        std::cout << "Left Tau Cmd: " << tau_cmd_leftarm << std::endl;
        std::cout << std::endl;

        // ===== Send Commands to Hardware =====
        KDL::JntArray tau_zero(NUM_JOINTS);

        // kdl_tau_to_mit(tau_zero, leftarm_state.mit_cmd);
        kdl_tau_to_mit(tau_cmd_leftarm, leftarm_state.mit_cmd);
        openarm_compl_ctrl::openarm_update(leftarm, leftarm_state, gleftarm_state);

        kdl_tau_to_mit(tau_zero, rightarm_state.mit_cmd);
        openarm_compl_ctrl::openarm_update(rightarm, rightarm_state, grightarm_state);

        // ===== Copy Sensor Readings Back =====
        for (int i = 0; i < NUM_JOINTS; ++i) {
            state.left.q[i]       = leftarm_state.q[i];
            state.left.qd[i]      = leftarm_state.qd[i];
            state.left.tau_mes[i] = leftarm_state.tau_mes[i];

            state.right.q[i]       = rightarm_state.q[i];
            state.right.qd[i]      = rightarm_state.qd[i];
            state.right.tau_mes[i] = rightarm_state.tau_mes[i];
        }
        state.left_gripper.q        = gleftarm_state.q;
        state.left_gripper.qd       = gleftarm_state.qd;
        state.left_gripper.tau_mes  = gleftarm_state.tau_mes;

        state.right_gripper.q        = grightarm_state.q;
        state.right_gripper.qd       = grightarm_state.qd;
        state.right_gripper.tau_mes  = grightarm_state.tau_mes;

        // Loop timing control
        while (now < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            now = std::chrono::high_resolution_clock::now();
        }
        while (deadline < now) {
            deadline += desired_loop_rate;
        }
    }

    // Shutdown - disable motors before tearing down ROS
    std::cout << "Disabling arm motors..." << std::endl;
    openarm_compl_ctrl::openarm_shutdown(leftarm);
    openarm_compl_ctrl::openarm_shutdown(rightarm);

    std::cout << "Shutting down node..." << std::endl;
    executor.cancel();
    rclcpp::shutdown();
    ros_thread.join();
    return 0;
}

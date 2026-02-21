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
#include "kdl/chainhdsolver_vereshchagin_fext.hpp"

#include "openarm_compl_ctrl/openarm_comm.hpp"
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>


#define LOG_INFO(node, msg, ...) RCLCPP_INFO(node->get_logger(), msg, ##__VA_ARGS__)
#define LOG_ERROR(node, msg, ...) RCLCPP_ERROR(node->get_logger(), msg, ##__VA_ARGS__)

#define LOG_INFO_S(node, expr) RCLCPP_INFO_STREAM((node)->get_logger(), expr)
#define LOG_ERROR_S(node, expr) RCLCPP_ERROR_STREAM((node)->get_logger(), expr)

#define NUM_JOINTS 7

std::atomic_bool shutting_down{false};

void signal_handler(int /*signum*/) {
    shutting_down.store(true);
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
                  double output_min, double output_max,
                  double derivative_lpf_cutoff,
                  double sample_rate,
                  double integral_decay_rate,
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
        // d_term_ = kd_ * raw_derivative;
        d_term_ = kd_ * derivative_filter_.update(raw_derivative);

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

    // assert(num_segments_left == num_joints_left && "Left arm has unexpected number of segments");
    assert(num_segments_right == num_joints_right && "Right arm has unexpected number of segments");
    
    LOG_INFO_S(node, "Left arm gravity in base frame: " << g_left_base);
    LOG_INFO_S(node, "Right arm gravity in base frame: " << g_right_base);

    // --------------------- State ----------------------

    State state;

    // --------------------- Vereshchagin Config ---------------------
    KDL::Twist g_left_arm(
        KDL::Vector(-g_left_base.x(), -g_left_base.y(), -g_left_base.z()),
        KDL::Vector::Zero()
    );
    KDL::Twist g_right_arm(
        KDL::Vector(-g_right_base.x(), -g_right_base.y(), -g_right_base.z()),
        KDL::Vector::Zero()
    );
    
    int num_constraints = 6;

    KDL::ChainHdSolver_Vereshchagin achd_solver_left(left_arm_chain, g_left_arm, num_constraints);

    KDL::JntArray ff_taus(num_joints_left);

    KDL::Wrenches f_ext(num_segments_left);
    for (size_t i = 0; i < f_ext.size(); ++i) {
        f_ext[i] = KDL::Wrench::Zero();
    }

    KDL::Jacobian alpha_left_world(num_constraints);
    KDL::JntArray beta_left(num_constraints);
    
    alpha_left_world.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    alpha_left_world.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0))); 
    alpha_left_world.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0))); 
    alpha_left_world.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0))); 
    alpha_left_world.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1))); 
           
    // transform world to respective arm base frame
    KDL::Frame f_left_base_inv = f_left_base.Inverse();

    KDL::Jacobian alpha_left(num_constraints);
    for (int i = 0; i < num_constraints; ++i) {
        alpha_left.setColumn(i, f_left_base_inv * alpha_left_world.getColumn(i));
    }

    KDL::ChainFkSolverVel_recursive fk_vel_solver_left(left_arm_chain);

    KDL::FrameVel left_ee_fvel;
    KDL::JntArrayVel left_q_qd(num_joints_left);

    fk_vel_solver_left.JntToCart(left_q_qd, left_ee_fvel);


    KDL::JntArray leftarm_q(NUM_JOINTS), leftarm_qd(NUM_JOINTS), leftarm_qdd(NUM_JOINTS);
    KDL::JntArray leftarm_tau_cmd(NUM_JOINTS);

    for (int i = 0; i < NUM_JOINTS; ++i) {
        leftarm_q(i) = state.left.q[i];
        leftarm_qd(i) = state.left.qd[i];
    }
    int r = achd_solver_left.CartToJnt(
                                    leftarm_q,
                                    leftarm_qd,
                                    leftarm_qdd,
                                    alpha_left, beta_left, 
                                    f_ext, ff_taus, 
                                    leftarm_tau_cmd);
    if (r < 0) {
        LOG_ERROR(node, "Failed to compute feedforward torques for left arm: %d", r);
        return -1;
    }

    ArmHardwareConfig left_arm_config = openarm_compl_ctrl::make_default_arm_config("can1");

    openarm::can::socket::OpenArm left_arm(left_arm_config.can_interface, true);
    
    if (!openarm_compl_ctrl::openarm_init(left_arm, left_arm_config)) {
        LOG_ERROR(node, "Failed to initialize left arm");
        return -1;
    }

    if (!openarm_compl_ctrl::openarm_start(left_arm)) {
        LOG_ERROR(node, "Failed to start left arm");
        return -1;
    }

    openarm_compl_ctrl::ArmState leftarm_state = {};
    leftarm_state.mit_cmd.resize(NUM_JOINTS);
    openarm_compl_ctrl::GripperState leftg_state = {};

    openarm::damiao_motor::MITParam zero_mit;
    leftg_state.mit_cmd = zero_mit;
    openarm_compl_ctrl::openarm_update(left_arm, leftarm_state, leftg_state);
    // copy back
    for (int i = 0; i < NUM_JOINTS; ++i) {
        state.left.q[i] = leftarm_state.q[i];
        state.left.qd[i] = leftarm_state.qd[i];
        state.left.tau_mes[i] = leftarm_state.tau_mes[i];
    }
    state.left_gripper.q = leftg_state.q;
    state.left_gripper.qd = leftg_state.qd;
    state.left_gripper.tau_mes = leftg_state.tau_mes;
 
    
    // PID controllers
    PIDController left_ee_lin_vel_x_pid(100.0, 0.0, 0.5, -100.0, 100.0, 10.0, 1000.0, 0.9, 0.0);
    PIDController left_ee_lin_vel_y_pid(0.0, 0.0, 0.0, -100.0, 100.0, 10.0, 1000.0, 0.9, 0.0);
    PIDController left_ee_lin_vel_z_pid(0.0, 0.0, 0.0, -100.0, 100.0, 10.0, 1000.0, 0.9, 0.0);

    PIDController left_ee_ang_vel_x_pid(0.0, 0.0, 0.0, -500.0, 500.0, 10.0, 1000.0, 0.9, 0.0);
    PIDController left_ee_ang_vel_y_pid(0.0, 0.0, 0.0, -500.0, 500.0, 10.0, 1000.0, 0.9, 0.0);
    PIDController left_ee_ang_vel_z_pid(0.0, 0.0, 0.0, -1000.0, 1000.0, 10.0, 1000.0, 0.9, 0.0);

    
    constexpr double TAU_MAX = 10.0;
    constexpr double DT      = 0.001; // 1000 Hz control loop

    LOG_INFO(node, "Starting control loop...");

    // DT-based loop timing
    auto desired_loop_rate = std::chrono::microseconds(static_cast<int>(DT * 1e6));
    auto now = std::chrono::high_resolution_clock::now();
    auto deadline = now + desired_loop_rate;
    

    while (!shutting_down.load()) {

        for (int i = 0; i < NUM_JOINTS; ++i) {
            leftarm_q(i) = state.left.q[i];
            leftarm_qd(i) = state.left.qd[i];
        }

        KDL::JntArrayVel left_q_qd(leftarm_q, leftarm_qd);
        fk_vel_solver_left.JntToCart(left_q_qd, left_ee_fvel);

        KDL::Frame left_ee_frame = left_ee_fvel.GetFrame();
        KDL::Twist left_ee_vel = left_ee_fvel.GetTwist();

        // Transform end-effector frame and velocity to world frame for constraint evaluation
        KDL::Frame left_ee_frame_world = f_left_base * left_ee_frame;
        KDL::Twist left_ee_vel_world = f_left_base * left_ee_vel;

        left_ee_fvel = KDL::FrameVel(left_ee_frame_world, left_ee_vel_world);

        std::cout << "Left EE Position (world): " << left_ee_frame_world.p << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        beta_left(0) = 0.0;
        beta_left(1) = 0.0;
        beta_left(2) = 0.0;
        beta_left(3) = 0.0;
        beta_left(4) = 0.0;
        beta_left(5) = 0.0;

        int r = achd_solver_left.CartToJnt(
                                leftarm_q,
                                leftarm_qd,
                                leftarm_qdd,
                                alpha_left, beta_left,
                                f_ext, ff_taus,
                                leftarm_tau_cmd);
        if (r < 0) {
            LOG_ERROR(node, "Vereshchagin CartToJnt failed in main loop: %d", r);
        }

        std::cout << "tau_cmd: " << leftarm_tau_cmd << std::endl;

        // clamp torques to max limits
        for (int i = 0; i < num_joints_left; ++i) {
            leftarm_tau_cmd(i) = std::clamp(leftarm_tau_cmd(i), -TAU_MAX, TAU_MAX);
        }

        // udpate state
        state.left_tau_cmd = leftarm_tau_cmd;
        state.left_ee_fvel = left_ee_fvel;

        robot_state->update(state);
        
        // update robot mit commands
        for (int i = 0; i < NUM_JOINTS; ++i) {
            leftarm_state.mit_cmd[i].tau = 0*leftarm_tau_cmd(i);
        }
        openarm_compl_ctrl::openarm_update(left_arm, leftarm_state, leftg_state);

        // copy back
        for (int i = 0; i < NUM_JOINTS; ++i) {
            state.left.q[i] = leftarm_state.q[i];
            state.left.qd[i] = leftarm_state.qd[i];
            state.left.tau_mes[i] = leftarm_state.tau_mes[i];
        }
        state.left_gripper.q = leftg_state.q;
        state.left_gripper.qd = leftg_state.qd;
        state.left_gripper.tau_mes = leftg_state.tau_mes;

        while (now < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            now = std::chrono::high_resolution_clock::now();
        }
        while (deadline < now) {
            deadline += desired_loop_rate;
        }
    }

    // shutdown - disable motors before tearing down ROS
    std::cout << "Disabling arm's motors..." << std::endl;
    openarm_compl_ctrl::openarm_shutdown(left_arm);

    std::cout << "Shutting down node..." << std::endl;
    executor.cancel();
    rclcpp::shutdown();
    ros_thread.join();
    return 0;
}

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <csignal>
#include <iomanip>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/kinfam_io.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/treefksolverpos_recursive.hpp"
#include "kdl/chainhdsolver_vereshchagin.hpp"
#include "kdl/chainhdsolver_vereshchagin_fixed_joint.hpp"

#define LOG_ERROR(node, msg, ...) RCLCPP_ERROR(node->get_logger(), msg, ##__VA_ARGS__)

std::atomic_bool shutting_down{false};
void signal_handler(int) { shutting_down.store(true); }

int main(int argc, char **argv)
{
    std::cout << std::fixed << std::setprecision(5);

    rclcpp::InitOptions init_options;
    init_options.shutdown_on_signal = false;
    rclcpp::init(argc, argv, init_options);
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto node = std::make_shared<rclcpp::Node>("achd_test_node");
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    std::thread ros_thread([&executor]() { executor.spin(); });

    // ---- URDF ----
    std::string urdf_path =
        ament_index_cpp::get_package_share_directory("openarm_compl_ctrl")
        + "/urdf/openarm_bimanual.urdf";

    KDL::Tree kdl_tree;
    if (!kdl_parser::treeFromFile(urdf_path, kdl_tree)) {
        LOG_ERROR(node, "Failed to parse URDF: %s", urdf_path.c_str());
        return -1;
    }

    // -----------------------------------------------------------------------
    // Chains:
    //   A: link0 -> link7        (7 movable, 0 fixed)  original baseline
    //   B: world -> left_finger  (7 movable, fixed prefix + suffix)
    //   C: link0 -> left_finger  (7 movable, fixed suffix only)
    //   D: world -> link7        (7 movable, fixed prefix only)  <-- KEY TEST
    //      D has same tip as A → same lever arm, same mass → tau MUST == solver1/2
    //      D has world root     → uses world-frame gravity and alpha (no rotation)
    // -----------------------------------------------------------------------
    KDL::Chain chain_A, chain_B, chain_C, chain_D;

    if (!kdl_tree.getChain("openarm_left_link0", "openarm_left_link7",       chain_A)) { LOG_ERROR(node, "Failed chain A"); return -1; }
    if (!kdl_tree.getChain("world",              "openarm_left_link7",        chain_B)) { LOG_ERROR(node, "Failed chain B"); return -1; }
    if (!kdl_tree.getChain("openarm_left_link0", "openarm_left_left_finger",  chain_C)) { LOG_ERROR(node, "Failed chain C"); return -1; }
    if (!kdl_tree.getChain("world",              "openarm_left_left_finger",  chain_D)) { LOG_ERROR(node, "Failed chain D"); return -1; }

    RCLCPP_INFO(node->get_logger(), "Chain A: %d joints, %d segments  (link0 -> link7)",
                chain_A.getNrOfJoints(), chain_A.getNrOfSegments());
    RCLCPP_INFO(node->get_logger(), "Chain B: %d joints, %d segments  (world -> link7)",
                chain_B.getNrOfJoints(), chain_B.getNrOfSegments());
    RCLCPP_INFO(node->get_logger(), "Chain C: %d joints, %d segments  (link0 -> left_finger)",
                chain_C.getNrOfJoints(), chain_C.getNrOfSegments());
    RCLCPP_INFO(node->get_logger(), "Chain D: %d joints, %d segments  (world -> left_finger)",
                chain_D.getNrOfJoints(), chain_D.getNrOfSegments());

    // ---- Gravity ----
    const KDL::Vector gravity_world(0.0, 0.0, -9.81);
    KDL::TreeFkSolverPos_recursive fk_solver(kdl_tree);
    KDL::JntArray q_tree(kdl_tree.getNrOfJoints());

    KDL::Frame F_world_link0;
    fk_solver.JntToCart(q_tree, F_world_link0, "openarm_left_link0");
    KDL::Vector g_link0 = F_world_link0.M.Inverse() * gravity_world;

    // Chains A, C rooted at link0
    KDL::Twist root_acc_link0(
        KDL::Vector(-g_link0.x(), -g_link0.y(), -g_link0.z()),
        KDL::Vector::Zero());

    // Chains B, D rooted at world
    KDL::Twist root_acc_world(
        KDL::Vector(-gravity_world.x(), -gravity_world.y(), -gravity_world.z()),
        KDL::Vector::Zero());

    // ---- Alpha ----
    const int nc = 6;

    // World-frame unit axes
    KDL::Jacobian alpha_world(nc);
    alpha_world.setColumn(0, KDL::Twist(KDL::Vector(1,0,0), KDL::Vector::Zero()));
    alpha_world.setColumn(1, KDL::Twist(KDL::Vector(0,1,0), KDL::Vector::Zero()));
    alpha_world.setColumn(2, KDL::Twist(KDL::Vector(0,0,1), KDL::Vector::Zero()));
    alpha_world.setColumn(3, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(1,0,0)));
    alpha_world.setColumn(4, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(0,1,0)));
    alpha_world.setColumn(5, KDL::Twist(KDL::Vector::Zero(), KDL::Vector(0,0,1)));

    // Rotated into link0 frame for chains A, C
    KDL::Jacobian alpha_link0(nc);
    KDL::Rotation R = F_world_link0.M.Inverse();
    for (int i = 0; i < nc; ++i)
        alpha_link0.setColumn(i, R * alpha_world.getColumn(i));

    // ---- Solvers ----
    KDL::ChainHdSolver_Vereshchagin        solver1(chain_A, root_acc_link0, nc);  // original, chain A
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint solver2(chain_A, root_acc_link0, nc);  // fixed_joint,   chain A  (sanity == solver1)
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint solver3(chain_B, root_acc_world, nc);  // fixed_joint,   chain B  (fixed prefix, same tip as A → must == solver1)
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint solver4(chain_C, root_acc_link0, nc);  // fixed_joint,   chain C  (fixed suffix)
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint solver5(chain_D, root_acc_world, nc);  // fixed_joint,   chain D  (fixed prefix+suffix)

    // ---- Arrays ----
    const int nj_A=chain_A.getNrOfJoints(), ns_A=chain_A.getNrOfSegments();
    const int nj_B=chain_B.getNrOfJoints(), ns_B=chain_B.getNrOfSegments();
    const int nj_C=chain_C.getNrOfJoints(), ns_C=chain_C.getNrOfSegments();
    const int nj_D=chain_D.getNrOfJoints(), ns_D=chain_D.getNrOfSegments();

    auto make_jnt = [](int n) { KDL::JntArray a(n); for(int i=0;i<n;++i) a(i)=0.0; return a; };
    auto make_w   = [](int n) { KDL::Wrenches w(n); for(auto& x:w) x=KDL::Wrench::Zero(); return w; };

    KDL::JntArray q_A=make_jnt(nj_A), qd_A=make_jnt(nj_A), ff_A=make_jnt(nj_A); KDL::Wrenches fext_A=make_w(ns_A);
    KDL::JntArray q_B=make_jnt(nj_B), qd_B=make_jnt(nj_B), ff_B=make_jnt(nj_B); KDL::Wrenches fext_B=make_w(ns_B);
    KDL::JntArray q_C=make_jnt(nj_C), qd_C=make_jnt(nj_C), ff_C=make_jnt(nj_C); KDL::Wrenches fext_C=make_w(ns_C);
    KDL::JntArray q_D=make_jnt(nj_D), qd_D=make_jnt(nj_D), ff_D=make_jnt(nj_D); KDL::Wrenches fext_D=make_w(ns_D);

    // ---- Beta test cases ----
    std::vector<std::array<double,6>> betas = {
        { 100,    0, 100,    0,    0,    0},
        {-100,    0, 100,    0,    0,    0},
        {   0,  100, 100,    0,    0,    0},
        {   0, -100, 100,    0,    0,    0},
        {   0,    0,   0, 1000,    0,    0},
        {   0,    0,   0,    0, 1000,    0},
        {   0,    0,   0,    0,    0, 1000},
    };

    // ---- Run ----
    for (const auto& b : betas)
    {
        KDL::JntArray beta(nc);
        for (int i = 0; i < nc; ++i) beta(i) = b[i];

        KDL::JntArray qdd1=make_jnt(nj_A), tau1=make_jnt(nj_A);
        KDL::JntArray qdd2=make_jnt(nj_A), tau2=make_jnt(nj_A);
        KDL::JntArray qdd3=make_jnt(nj_B), tau3=make_jnt(nj_B);
        KDL::JntArray qdd4=make_jnt(nj_C), tau4=make_jnt(nj_C);
        KDL::JntArray qdd5=make_jnt(nj_D), tau5=make_jnt(nj_D);

        int r1 = solver1.CartToJnt(q_A, qd_A, qdd1, alpha_link0, beta, fext_A, ff_A, tau1);
        int r2 = solver2.CartToJnt(q_A, qd_A, qdd2, alpha_link0, beta, fext_A, ff_A, tau2);
        int r3 = solver3.CartToJnt(q_B, qd_B, qdd3, alpha_world,  beta, fext_B, ff_B, tau3);
        int r4 = solver4.CartToJnt(q_C, qd_C, qdd4, alpha_link0, beta, fext_C, ff_C, tau4);
        int r5 = solver5.CartToJnt(q_D, qd_D, qdd5, alpha_world,  beta, fext_D, ff_D, tau5);

        std::cout << "beta: " << beta << "\n";
        std::cout << "  solver1 (orig,   A link0->link7):         " << (r1==0 ? tau1 : make_jnt(nj_A)) << "\n";
        std::cout << "  solver2 (fixed_joint, A link0->link7):         " << (r2==0 ? tau2 : make_jnt(nj_A)) << "\n";
        std::cout << "  solver3 (fixed_joint, B world->link7):         " << (r3==0 ? tau3 : make_jnt(nj_B)) << "\n";
        std::cout << "  solver4 (fixed_joint, C link0->finger):        " << (r4==0 ? tau4 : make_jnt(nj_C)) << "\n";
        std::cout << "  solver5 (fixed_joint, D world->finger):        " << (r5==0 ? tau5 : make_jnt(nj_D)) << "\n";
        std::cout << "\n";
    }

    executor.cancel();
    rclcpp::shutdown();
    ros_thread.join();
    return 0;
}

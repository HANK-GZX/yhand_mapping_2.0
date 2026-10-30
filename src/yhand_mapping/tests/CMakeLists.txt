# Keep test registration beside the tests with per-test timeouts and env.
function(add_mapping_pytest name timeout)
  ament_add_pytest_test(${name}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_${name}.py"
    TIMEOUT ${timeout}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    ENV "PYTHONDONTWRITEBYTECODE=1" ${ARGN}
  )
endfunction()

# Data acquisition, input handling and hardware-free control checks.
add_mapping_pytest(manus_action_one 30)
add_mapping_pytest(manus_action_two 30)
add_mapping_pytest(manus_action_three 30)
add_mapping_pytest(motor_mapping 90)
add_mapping_pytest(workspace_boundary_collect 90 "MPLBACKEND=Agg")
add_mapping_pytest(data_collection_extensor 30)
add_mapping_pytest(pretighten_shutdown 90)
# Includes 54 real DDS cases backed by a fake SDK driver.
add_mapping_pytest(left_little_q_to_motor_test 300)
add_mapping_pytest(left_thumb_q_to_motor 30)
add_mapping_pytest(thumb_combo_policy 60 "ROS_LOG_DIR=/tmp/yhand_combo_test_ros")

# Current q-to-motor models and their shared inference/coordinate contracts.
add_mapping_pytest(left_hand_q_to_motor_mlp 30 "ROS_LOG_DIR=/tmp/yhand_mlp_test_ros")
add_mapping_pytest(monotonic_extensor 30 "ROS_LOG_DIR=/tmp/yhand_monotonic_test_ros")
add_mapping_pytest(monotonic_sides 30 "ROS_LOG_DIR=/tmp/yhand_monotonic_sides_test_ros")
add_mapping_pytest(middle_mlp 30 "ROS_LOG_DIR=/tmp/yhand_middle_mlp_test_ros")
add_mapping_pytest(ring_mlp 30 "ROS_LOG_DIR=/tmp/yhand_ring_mlp_test_ros")
add_mapping_pytest(thumb_coordination_mlp 30 "ROS_LOG_DIR=/tmp/yhand_thumb_coordination_ros")

# Inverse mapping and recording.
add_mapping_pytest(left_thumb_motor_to_q_estimator 30)
add_mapping_pytest(left_hand_motor_to_q_estimator 30)
add_mapping_pytest(motor_to_q_logging 30 "MPLCONFIGDIR=/tmp/yhand_inverse_logging_tests")

cmake_minimum_required(VERSION 3.15)
project(yhand_mapping)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Select script roots before recursing: collection output must not be scanned
# or registered with CONFIGURE_DEPENDS. Use explicit files for both install modes.
file(GLOB MAPPING_SCRIPT_ENTRIES CONFIGURE_DEPENDS
  LIST_DIRECTORIES TRUE
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "scripts/*"
)
list(FILTER MAPPING_SCRIPT_ENTRIES EXCLUDE REGEX
  [[^scripts/(motor_test_logs|\.venv|venv|__pycache__|wandb)(/|$)]]
)
set(MAPPING_RESOURCES)
foreach(SCRIPT_ENTRY IN LISTS MAPPING_SCRIPT_ENTRIES)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT_ENTRY}")
    file(GLOB_RECURSE SCRIPT_RESOURCES CONFIGURE_DEPENDS
      LIST_DIRECTORIES FALSE
      RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
      "${SCRIPT_ENTRY}/*"
    )
    list(APPEND MAPPING_RESOURCES ${SCRIPT_RESOURCES})
  else()
    list(APPEND MAPPING_RESOURCES "${SCRIPT_ENTRY}")
  endif()
endforeach()

# Keep the active workspace models needed by data collection in copied installs.
# Raw recordings, plots, checkpoints and historical models remain local output.
file(GLOB MAPPING_BOUNDARY_MODELS CONFIGURE_DEPENDS
  LIST_DIRECTORIES FALSE
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "scripts/motor_test_logs/workspace_boundaries/left/boundary_models.json"
  "scripts/motor_test_logs/workspace_boundaries/right/boundary_models.json"
)
list(APPEND MAPPING_RESOURCES ${MAPPING_BOUNDARY_MODELS})
list(FILTER MAPPING_RESOURCES EXCLUDE REGEX
  [[(^|/)(\.venv|venv|__pycache__|wandb)(/|$)|\.py[co]$]]
)
foreach(RESOURCE IN LISTS MAPPING_RESOURCES)
  get_filename_component(RESOURCE_DIR "${RESOURCE}" DIRECTORY)
  if(RESOURCE MATCHES "\\.sh$")
    install(PROGRAMS "${RESOURCE}" DESTINATION share/${PROJECT_NAME}/${RESOURCE_DIR})
  else()
    install(FILES "${RESOURCE}" DESTINATION share/${PROJECT_NAME}/${RESOURCE_DIR})
  endif()
endforeach()

file(GLOB MAPPING_DOCUMENTS CONFIGURE_DEPENDS "*.md" "*.xlsx")
install(FILES LICENSE ${MAPPING_DOCUMENTS}
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(manus_action_one tests/test_manus_action_one.py
    TIMEOUT 30
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(manus_action_three tests/test_manus_action_three.py
    TIMEOUT 30
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(manus_action_two tests/test_manus_action_two.py
    TIMEOUT 30
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(motor_mapping tests/test_motor_mapping.py
    TIMEOUT 90
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(workspace_boundary_collect tests/test_workspace_boundary_collect.py
    TIMEOUT 90
    ENV "PYTHONDONTWRITEBYTECODE=1" "MPLBACKEND=Agg"
  )
  ament_add_pytest_test(data_collection_extensor tests/test_data_collection_extensor.py
    TIMEOUT 30
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(pretighten_shutdown tests/test_pretighten_shutdown.py
    TIMEOUT 90
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(left_little_q_to_motor_test tests/test_left_little_q_to_motor_test.py
    TIMEOUT 120
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
  ament_add_pytest_test(left_thumb_q_to_motor tests/test_left_thumb_q_to_motor.py
    TIMEOUT 30
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
endif()

ament_package()

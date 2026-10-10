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

# Explicitly filter resource paths so copy and ament symlink installs agree.
# Directory exclusion patterns alone do not filter nested files in symlink mode.
file(GLOB_RECURSE MAPPING_RESOURCES CONFIGURE_DEPENDS
  LIST_DIRECTORIES FALSE
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "scripts/*"
)
list(FILTER MAPPING_RESOURCES EXCLUDE REGEX
  [[(^|/)(\.venv|__pycache__|wandb)(/|$)|\.pyc$]]
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
  ament_add_pytest_test(pretighten_shutdown tests/test_pretighten_shutdown.py
    TIMEOUT 90
    ENV "PYTHONDONTWRITEBYTECODE=1"
  )
endif()

ament_package()

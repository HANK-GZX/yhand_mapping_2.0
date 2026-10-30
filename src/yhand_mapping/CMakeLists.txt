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
  add_subdirectory(tests)
endif()

ament_package()

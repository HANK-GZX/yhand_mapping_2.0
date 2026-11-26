cmake_minimum_required(VERSION 3.15)
project(yhand_mapping)

find_package(ament_cmake REQUIRED)

# Build the SDK demo with the mapping launch's configuration and feedback interface.
# Apply the adaptation to a build-tree copy of the upstream source.
set(YHAND_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../devices_pkg/sdk")
set(M2_DEMO_SOURCE "${YHAND_SDK_ROOT}/src/Project/Y_Hand_M2/out/Y_Hand_M2_Out.cpp")
if(EXISTS "${M2_DEMO_SOURCE}")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(SDK_LIB_ARCH x86_64)
    set(SDK_THIRD_ARCH x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(SDK_LIB_ARCH arch64)
    set(SDK_THIRD_ARCH arm64)
  else()
    message(FATAL_ERROR "Y_Hand_M2_Demo SDK libraries are unavailable for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(SDK_INCLUDE "${YHAND_SDK_ROOT}/inc")
  set(SDK_THIRD_PARTY "${YHAND_SDK_ROOT}/third_party/lib/install-${SDK_THIRD_ARCH}")
  set(M2_DEMO_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/native/y_hand_m2_demo.patch")
  set(M2_DEMO_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Y_Hand_M2_Demo.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${M2_DEMO_SOURCE}" "${M2_DEMO_PATCH}")
  find_program(PATCH_EXECUTABLE patch)
  if(NOT PATCH_EXECUTABLE)
    message(FATAL_ERROR "The patch executable is required to build the SDK Y_Hand_M2_Demo")
  endif()
  execute_process(
    COMMAND "${PATCH_EXECUTABLE}" --batch --silent --ignore-whitespace --fuzz=0
            -o "${M2_DEMO_OUTPUT}" "${M2_DEMO_SOURCE}"
    INPUT_FILE "${M2_DEMO_PATCH}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE M2_PATCH_RESULT
  )
  if(NOT M2_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not adapt SDK Y_Hand_M2_Demo. Check the SDK source revision.")
  endif()

  add_executable(Y_Hand_M2_Demo "${M2_DEMO_OUTPUT}")
  target_compile_features(Y_Hand_M2_Demo PRIVATE cxx_std_17)
  target_include_directories(Y_Hand_M2_Demo PRIVATE
    "${SDK_INCLUDE}/HARDWARE_TOP"
    "${SDK_INCLUDE}/Z_SYS_SRC/SYS"
    "${SDK_INCLUDE}/Z_SYS_SRC/CAN"
    "${SDK_INCLUDE}/Z_SYS_SRC/UDP"
    "${SDK_INCLUDE}/Z_SYS_SRC/Socket_CAN"
    "${SDK_INCLUDE}/Motors/Motor"
    "${SDK_INCLUDE}/Switch_Boards/Switch_Board"
    "${SDK_INCLUDE}/Switch_Boards/Switch_Board/mcu/stm32h743"
    "${SDK_INCLUDE}/Misc/Auto_Set_Id"
    "${SDK_INCLUDE}/Misc/Led_Device"
    "${SDK_THIRD_PARTY}/include"
  )
  target_link_directories(Y_Hand_M2_Demo PRIVATE
    "${YHAND_SDK_ROOT}/lib/${SDK_LIB_ARCH}" "${SDK_THIRD_PARTY}/lib")
  target_link_libraries(Y_Hand_M2_Demo PRIVATE
    -Wl,--no-as-needed HARDWARE_TOP Switch_Board Motor_Device Auto_Set_Id Led_Device
    yaml-cpp spdlogs pthread -Wl,--as-needed)
  install(TARGETS Y_Hand_M2_Demo RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()

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

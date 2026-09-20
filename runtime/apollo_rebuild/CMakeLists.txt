# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

cmake_minimum_required(VERSION 3.20)
project(ApolloCodeBase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(APOLLO_CODE_BASE_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
    CACHE PATH "Path to ONNX Runtime root")
option(APOLLO_STATIC_YAML_CPP "Prefer the static yaml-cpp archive when available" ON)
option(APOLLO_STATIC_CXX_RUNTIME "Link libstdc++ and libgcc statically into Apollo" ON)

find_package(yaml-cpp REQUIRED)

set(APOLLO_YAML_CPP_PACKAGE_TARGET yaml-cpp)
if(TARGET yaml-cpp::yaml-cpp)
    set(APOLLO_YAML_CPP_PACKAGE_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(APOLLO_YAML_CPP_PACKAGE_TARGET yaml-cpp)
endif()

set(APOLLO_YAML_CPP_LINK "${APOLLO_YAML_CPP_PACKAGE_TARGET}")
if(APOLLO_STATIC_YAML_CPP)
    find_library(APOLLO_YAML_CPP_STATIC_LIBRARY NAMES libyaml-cpp.a)
    if(APOLLO_YAML_CPP_STATIC_LIBRARY)
        add_library(apollo_yaml_cpp_static UNKNOWN IMPORTED)
        set_target_properties(apollo_yaml_cpp_static PROPERTIES
            IMPORTED_LOCATION "${APOLLO_YAML_CPP_STATIC_LIBRARY}"
        )
        if(TARGET ${APOLLO_YAML_CPP_PACKAGE_TARGET})
            foreach(prop IN ITEMS
                INTERFACE_INCLUDE_DIRECTORIES
                INTERFACE_COMPILE_DEFINITIONS
                INTERFACE_COMPILE_OPTIONS
            )
                get_target_property(prop_value ${APOLLO_YAML_CPP_PACKAGE_TARGET} ${prop})
                if(prop_value)
                    set_target_properties(apollo_yaml_cpp_static PROPERTIES
                        ${prop} "${prop_value}"
                    )
                endif()
            endforeach()
        endif()
        set(APOLLO_YAML_CPP_LINK apollo_yaml_cpp_static)
        message(STATUS "Using static yaml-cpp: ${APOLLO_YAML_CPP_STATIC_LIBRARY}")
    else()
        message(STATUS "Static yaml-cpp archive not found; using ${APOLLO_YAML_CPP_PACKAGE_TARGET}")
    endif()
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must point to the ONNX Runtime distribution root")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0")
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found under ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIBRARY}")
endif()

add_library(agent_app
    src/app/agent_app.cc
    src/app/runtime_config.cc
)

add_library(robot_core
    src/robot/t1_robot_model.cc
)

add_library(world_core
    src/world/frame_normalizer.cc
    src/world/play_mode.cc
    src/world/world_state.cc
)

add_library(comm_core
    src/comm/team_comm_codec.cc
    src/comm/team_comm_manager.cc
)

add_library(decision_core
    src/decision/blackboard.cc
    src/decision/behavior_tree.cc
    src/decision/decision_manager.cc
    src/decision/formation.cc
    src/decision/role_behaviors.cc
    src/decision/role_manager_core.cc
    src/decision/role_manager_assignment.cc
    src/decision/walk_planner.cc
)

add_library(behavior_core
    src/behavior/dynamic_pass_runner.cc
    src/behavior/getup_runner.cc
    src/behavior/keyframe_runner.cc
    src/behavior/kick_execution_profile.cc
    src/behavior/kick_residual_runner.cc
    src/behavior/learned_kick_runner.cc
    src/behavior/motion_manager.cc
    src/behavior/onnx_session.cc
    src/behavior/procedural_kick_runner.cc
    src/behavior/walk_runner.cc
)

add_library(server_core
    src/server/action_encoder.cc
    src/server/tcp_lpm_client.cc
    src/server/perception_parser.cc
)

target_include_directories(agent_app
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(robot_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(world_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(comm_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(server_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(decision_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(behavior_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_options(agent_app
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(robot_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(world_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(comm_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(server_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(decision_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(behavior_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_libraries(world_core PUBLIC robot_core)
target_link_libraries(comm_core PUBLIC world_core)
target_link_libraries(server_core PUBLIC robot_core)
target_link_libraries(decision_core PUBLIC world_core)
target_link_libraries(behavior_core PUBLIC robot_core world_core decision_core "${APOLLO_YAML_CPP_LINK}" "${ONNXRUNTIME_LIBRARY}")
set_target_properties(behavior_core PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
target_link_libraries(agent_app PRIVATE decision_core behavior_core server_core comm_core)

add_executable(ApolloCodeBase main.cc)
target_link_libraries(ApolloCodeBase PRIVATE agent_app)
set_target_properties(ApolloCodeBase PROPERTIES
    BUILD_RPATH "\$ORIGIN/libs;${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "\$ORIGIN/libs"
)
if(APOLLO_STATIC_CXX_RUNTIME AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(ApolloCodeBase PRIVATE -static-libstdc++ -static-libgcc)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(goal_kick_behavior_test
        tests/goal_kick_behavior_test.cc
    )
    target_link_libraries(goal_kick_behavior_test PRIVATE decision_core)
    add_test(NAME goal_kick_behavior_test COMMAND goal_kick_behavior_test)

    add_executable(discrete_ball_action_test
        tests/discrete_ball_action_test.cc
    )
    target_link_libraries(discrete_ball_action_test PRIVATE decision_core)
    add_test(NAME discrete_ball_action_test COMMAND discrete_ball_action_test)

    add_executable(dynamic_pass_runner_test
        tests/dynamic_pass_runner_test.cc
    )
    target_link_libraries(dynamic_pass_runner_test PRIVATE behavior_core)
    add_test(NAME dynamic_pass_runner_test COMMAND dynamic_pass_runner_test)

    add_executable(runtime_config_test
        tests/runtime_config_test.cc
        src/app/runtime_config.cc
    )
    target_include_directories(runtime_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME runtime_config_test COMMAND runtime_config_test)

    add_executable(run_policy_reflection_test
        tests/run_policy_reflection_test.cc
    )
    target_link_libraries(run_policy_reflection_test PRIVATE world_core)
    target_include_directories(run_policy_reflection_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME run_policy_reflection_test COMMAND run_policy_reflection_test)

    add_executable(kick_execution_profile_test
        tests/kick_execution_profile_test.cc
        src/behavior/kick_execution_profile.cc
    )
    target_link_libraries(kick_execution_profile_test PRIVATE world_core)
    target_include_directories(kick_execution_profile_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME kick_execution_profile_test COMMAND kick_execution_profile_test)

    add_executable(kick_residual_runner_test
        tests/kick_residual_runner_test.cc
        src/behavior/kick_residual_runner.cc
    )
    target_link_libraries(
        kick_residual_runner_test
        PRIVATE world_core "${APOLLO_YAML_CPP_LINK}"
    )
    target_include_directories(kick_residual_runner_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME kick_residual_runner_test COMMAND kick_residual_runner_test)

    add_executable(procedural_kick_runner_test
        tests/procedural_kick_runner_test.cc
        src/behavior/procedural_kick_runner.cc
    )
    target_link_libraries(
        procedural_kick_runner_test
        PRIVATE world_core "${APOLLO_YAML_CPP_LINK}"
    )
    target_include_directories(procedural_kick_runner_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME procedural_kick_runner_test COMMAND procedural_kick_runner_test)

    add_executable(learned_kick_runner_test
        tests/learned_kick_runner_test.cc
    )
    target_link_libraries(learned_kick_runner_test PRIVATE behavior_core)
    target_include_directories(learned_kick_runner_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME learned_kick_runner_test COMMAND learned_kick_runner_test)

    # The test sources use assert(); Release must not compile their checks out.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        foreach(test_target IN ITEMS
            goal_kick_behavior_test discrete_ball_action_test
            dynamic_pass_runner_test runtime_config_test
            run_policy_reflection_test kick_execution_profile_test
            kick_residual_runner_test procedural_kick_runner_test
            learned_kick_runner_test)
            target_compile_options(${test_target} PRIVATE -UNDEBUG)
        endforeach()
    endif()
endif()

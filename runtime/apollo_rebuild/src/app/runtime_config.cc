// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/app/runtime_config.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace app {

namespace {

bool is_flag(const std::string& arg, const char* short_flag, const char* long_flag) {
    return arg == short_flag || arg == long_flag;
}

}  // namespace

RuntimeConfig RuntimeConfig::from_args(int argc, char* argv[]) {
    RuntimeConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};

        auto require_value = [&](const char* flag_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string{"Missing value for "} + flag_name);
            }
            return argv[++i];
        };

        if (is_flag(arg, "-t", "--team")) {
            config.team_name = require_value("--team");
        } else if (is_flag(arg, "-n", "--player-number")) {
            config.player_number = std::stoi(require_value("--player-number"));
        } else if (is_flag(arg, "-h", "--host")) {
            config.host = require_value("--host");
        } else if (is_flag(arg, "-p", "--port")) {
            config.port = std::stoi(require_value("--port"));
        } else if (arg == "--asset-root") {
            config.asset_root = require_value("--asset-root");
        } else if (arg == "--config-root") {
            config.config_root = require_value("--config-root");
        } else if (arg == "--log-level") {
            config.log_level = require_value("--log-level");
        } else if (arg == "--status-interval") {
            config.status_interval = std::stoull(
                require_value("--status-interval"));
        } else if (arg == "--training-telemetry-interval") {
            config.training_telemetry_interval = std::stoull(
                require_value("--training-telemetry-interval"));
        } else if (arg == "--enable-dynamic-pass") {
            config.enable_dynamic_pass = true;
        } else if (arg == "--disable-dynamic-pass") {
            config.enable_dynamic_pass = false;
        } else if (arg == "--dynamic-pass-force-rollout") {
            config.dynamic_pass_forced_rollout_id = std::stoi(
                require_value("--dynamic-pass-force-rollout"));
        } else if (arg == "--enable-goalkeeper-intercept") {
            config.enable_goalkeeper_intercept = true;
        } else if (arg == "--disable-goalkeeper-intercept") {
            config.enable_goalkeeper_intercept = false;
        } else if (arg == "--enable-parameterized-kick") {
            config.enable_parameterized_kick = true;
        } else if (arg == "--disable-parameterized-kick") {
            config.enable_parameterized_kick = false;
        } else if (arg == "--enable-fast-walk") {
            config.enable_fast_walk = true;
        } else if (arg == "--disable-fast-walk") {
            config.enable_fast_walk = false;
        } else if (arg == "--fast-walk-model") {
            config.fast_walk_model = require_value("--fast-walk-model");
        } else if (arg == "--fast-walk-yaw-bias") {
            config.fast_walk_yaw_bias_rad_s = std::stod(
                require_value("--fast-walk-yaw-bias"));
        } else if (arg == "--enable-rapid-turn") {
            config.enable_rapid_turn = true;
        } else if (arg == "--disable-rapid-turn") {
            config.enable_rapid_turn = false;
        } else if (arg == "--rapid-turn-model") {
            config.rapid_turn_model = require_value("--rapid-turn-model");
        } else if (arg == "--enable-learned-kick") {
            config.enable_learned_kick = true;
        } else if (arg == "--disable-learned-kick") {
            config.enable_learned_kick = false;
        } else if (arg == "--shadow-learned-kick") {
            config.shadow_learned_kick = true;
        } else if (arg == "--disable-learned-kick-shadow") {
            config.shadow_learned_kick = false;
        } else if (arg == "--learned-kick-model") {
            config.learned_kick_model = require_value("--learned-kick-model");
        }
    }

    if (config.enable_fast_walk && config.fast_walk_model.empty()) {
        throw std::invalid_argument(
            "--enable-fast-walk requires --fast-walk-model");
    }
    if (!std::isfinite(config.fast_walk_yaw_bias_rad_s) ||
        std::abs(config.fast_walk_yaw_bias_rad_s) > 0.2) {
        throw std::invalid_argument(
            "--fast-walk-yaw-bias must be finite and within [-0.2, 0.2]");
    }
    if (config.enable_rapid_turn && config.rapid_turn_model.empty()) {
        throw std::invalid_argument(
            "--enable-rapid-turn requires --rapid-turn-model");
    }
    if (config.enable_learned_kick && config.shadow_learned_kick) {
        throw std::invalid_argument(
            "learned kick active and shadow modes are mutually exclusive");
    }
    if ((config.enable_learned_kick || config.shadow_learned_kick) &&
        !config.enable_parameterized_kick) {
        throw std::invalid_argument(
            "learned kick modes require --enable-parameterized-kick");
    }
    if ((config.enable_learned_kick || config.shadow_learned_kick) &&
        config.learned_kick_model.empty()) {
        throw std::invalid_argument(
            "learned kick modes require --learned-kick-model");
    }

    return config;
}

}  // namespace app

// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#include "src/app/runtime_config.h"

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
        } else if (arg == "--enable-fast-walk") {
            config.enable_fast_walk = true;
        } else if (arg == "--fast-walk-model") {
            config.fast_walk_model = require_value("--fast-walk-model");
        } else if (arg == "--enable-rapid-turn") {
            config.enable_rapid_turn = true;
        } else if (arg == "--rapid-turn-model") {
            config.rapid_turn_model = require_value("--rapid-turn-model");
        }
    }

    if (config.enable_fast_walk && config.fast_walk_model.empty()) {
        throw std::invalid_argument(
            "--enable-fast-walk requires --fast-walk-model");
    }
    if (config.enable_rapid_turn && config.rapid_turn_model.empty()) {
        throw std::invalid_argument(
            "--enable-rapid-turn requires --rapid-turn-model");
    }

    return config;
}

}  // namespace app

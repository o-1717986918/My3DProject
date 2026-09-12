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
        } else if (arg == "--enable-dynamic-pass") {
            config.enable_dynamic_pass = true;
        } else if (arg == "--disable-dynamic-pass") {
            config.enable_dynamic_pass = false;
        } else if (arg == "--enable-goalkeeper-intercept") {
            config.enable_goalkeeper_intercept = true;
        } else if (arg == "--disable-goalkeeper-intercept") {
            config.enable_goalkeeper_intercept = false;
        }
    }

    return config;
}

}  // namespace app

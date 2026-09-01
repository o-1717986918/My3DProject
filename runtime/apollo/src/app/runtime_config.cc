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
        } else if (arg == "--max-cycles") {
            const auto value = std::stoull(require_value("--max-cycles"));
            if (value == 0U) {
                throw std::invalid_argument("--max-cycles must be positive");
            }
            config.max_cycles = static_cast<std::size_t>(value);
        } else if (arg == "--status-interval") {
            const auto value = std::stoull(require_value("--status-interval"));
            if (value == 0U) {
                throw std::invalid_argument("--status-interval must be positive");
            }
            config.status_interval_cycles = static_cast<std::size_t>(value);
        } else if (arg == "--disable-pass-strategy") {
            config.enable_pass_strategy = false;
        } else if (arg == "--enable-pass-strategy") {
            config.enable_pass_strategy = true;
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
        }
    }

    if (config.player_number < 1 || config.player_number > 7) {
        throw std::invalid_argument("--player-number must be in 1..7");
    }
    if (config.enable_fast_walk && config.fast_walk_model.empty()) {
        throw std::invalid_argument(
            "--enable-fast-walk requires --fast-walk-model");
    }

    return config;
}

}  // namespace app

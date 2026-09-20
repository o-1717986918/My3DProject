// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Hong Kong University of Science and Technology (Guangzhou), Humanoid Computing & Learning Lab.

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace app {

/// Command-line and runtime paths for a single agent process.
struct RuntimeConfig {
    std::string team_name{"Apollo3Drelease"};
    int player_number{1};
    std::string host{"127.0.0.1"};
    int port{60000};
    std::string asset_root{"assets"};
    std::string config_root{"config"};
    std::string log_level{"info"};
    std::size_t status_interval{0};
    // Opt-in, per-agent server-observed motion frames for offline training.
    std::size_t training_telemetry_interval{0};
    bool enable_dynamic_pass{true};
    int dynamic_pass_forced_rollout_id{-1};
    bool enable_goalkeeper_intercept{true};
    bool enable_parameterized_kick{false};
    bool enable_fast_walk{false};
    std::string fast_walk_model;
    // Experimental phase-v2 straight-line yaw compensation; zero preserves
    // the frozen runtime behavior until a real-match comparison is complete.
    double fast_walk_yaw_bias_rad_s{0.0};
    bool enable_rapid_turn{false};
    std::string rapid_turn_model;
    bool enable_learned_kick{false};
    bool shadow_learned_kick{false};
    std::string learned_kick_model;

    /// Parses supported command-line options and preserves unspecified defaults.
    static RuntimeConfig from_args(int argc, char* argv[]);

    /// Resolves an asset path against the configured or project asset root.
    std::filesystem::path resolve_asset_path(const std::string& relative_asset_path) const {
        const std::filesystem::path root(asset_root);
        if (root.is_absolute()) {
            return root / relative_asset_path;
        }
#ifdef APOLLO_CODE_BASE_PROJECT_SOURCE_DIR
        return std::filesystem::path(APOLLO_CODE_BASE_PROJECT_SOURCE_DIR) / root / relative_asset_path;
#else
        return std::filesystem::current_path() / root / relative_asset_path;
#endif
    }
};

}  // namespace app

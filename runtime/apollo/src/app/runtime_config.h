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
    /// Optional bounded run used by integration and competition-soak tests.
    /// Zero preserves the upstream run-until-disconnect behavior.
    std::size_t max_cycles{0U};
    /// Emit one machine-readable status line at this cycle interval.
    /// Zero keeps the competition runtime silent.
    std::size_t status_interval_cycles{0U};
    /// Enables the bounded one-step pass planner and pass-intent protocol.
    bool enable_pass_strategy{true};
    /// Enables coordinated open-play TeamTactics duties.  Disabling this is
    /// an ablation/debug surface; role assignment, formations and restart
    /// legality remain active.
    bool enable_team_tactics{true};
    /// Enables the experimental target/speed-conditioned contact macro.
    /// The validated fixed forward contact remains the default until server
    /// calibration passes the R1 promotion gate.
    bool enable_parameterized_kick{false};
    /// Enables full-body ownership by the experimental phase-v2 fast-walk
    /// actor on supported long, forward travel commands. Unsupported commands
    /// retain the original stable walk and get-up paths.
    bool enable_fast_walk{false};
    /// Explicit external-local ONNX path. The restricted research model is
    /// never bundled into the competition asset tree.
    std::string fast_walk_model;
    /// Enables the direction-routed rapid-turn actor for pure-yaw commands.
    /// Negative yaw is served by exact observation/action reflection rather
    /// than an unvalidated weak-direction policy branch.
    bool enable_rapid_turn{false};
    /// Explicit external-local phase-v2 run-policy ONNX path.
    std::string rapid_turn_model;
    /// Runs an external kick_policy_v3 actor as the active target-pass
    /// controller. This remains explicit opt-in until a candidate is promoted.
    bool enable_learned_kick{false};
    /// Evaluates kick_policy_v3 without granting it joint ownership. Shadow
    /// inference exercises the deployment contract while fallback stays live.
    bool shadow_learned_kick{false};
    /// Explicit external-local ONNX path. Training output existence alone
    /// never changes the packaged competition behavior.
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

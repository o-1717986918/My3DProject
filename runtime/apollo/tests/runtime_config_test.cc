// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/app/runtime_config.h"
#include "src/strategy/action_capability.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

app::RuntimeConfig parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    return app::RuntimeConfig::from_args(static_cast<int>(argv.size()), argv.data());
}

bool throws_invalid_argument(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    const app::RuntimeConfig config = parse({
        "ApolloCodeBase", "--team", "My3D", "--player-number", "7",
        "--host", "localhost", "--port", "61000", "--max-cycles", "800",
        "--status-interval", "20",
        "--disable-pass-strategy", "--enable-parameterized-kick",
        "--enable-fast-walk", "--fast-walk-model", "/tmp/fast-walk.onnx",
        "--shadow-learned-kick", "--learned-kick-model",
        "/tmp/learned-kick.onnx",
    });

    if (config.team_name != "My3D" || config.player_number != 7 ||
        config.host != "localhost" || config.port != 61000 ||
        config.max_cycles != 800U || config.status_interval_cycles != 20U ||
        config.enable_pass_strategy || !config.enable_parameterized_kick ||
        !config.enable_fast_walk ||
        config.fast_walk_model != "/tmp/fast-walk.onnx" ||
        config.enable_learned_kick || !config.shadow_learned_kick ||
        config.learned_kick_model != "/tmp/learned-kick.onnx") {
        std::cerr << "valid runtime arguments were not parsed correctly\n";
        return 1;
    }

    const app::RuntimeConfig enabled = parse({
        "ApolloCodeBase", "--disable-pass-strategy", "--enable-pass-strategy"});
    if (!enabled.enable_pass_strategy) {
        std::cerr << "pass strategy could not be re-enabled\n";
        return 1;
    }

    const app::RuntimeConfig safe_default = parse({"ApolloCodeBase"});
    if (safe_default.enable_parameterized_kick) {
        std::cerr << "experimental parameterized kick was enabled by default\n";
        return 1;
    }
    if (safe_default.enable_fast_walk || !safe_default.fast_walk_model.empty() ||
        safe_default.enable_learned_kick ||
        safe_default.shadow_learned_kick ||
        !safe_default.learned_kick_model.empty()) {
        std::cerr << "experimental fast walk was enabled by default\n";
        return 1;
    }

    const app::RuntimeConfig kick_disabled = parse({
        "ApolloCodeBase", "--enable-parameterized-kick",
        "--disable-parameterized-kick"});
    if (kick_disabled.enable_parameterized_kick) {
        std::cerr << "parameterized kick could not be disabled\n";
        return 1;
    }

    const strategy::ActionCapabilityRegistry safe_capabilities(false);
    const strategy::CooperativeAction pass{};
    if (safe_capabilities.state(strategy::SkillCapability::TargetedPass) !=
            strategy::CapabilityState::Unavailable ||
        safe_capabilities.state(strategy::SkillCapability::DribbleTouch) !=
            strategy::CapabilityState::Unavailable ||
        safe_capabilities.executable(pass, 0.0)) {
        std::cerr << "disabled target kick capability was executable\n";
        return 1;
    }
    const strategy::ActionCapabilityRegistry experimental_capabilities(true);
    if (experimental_capabilities.state(strategy::SkillCapability::TargetedPass) !=
            strategy::CapabilityState::Experimental ||
        experimental_capabilities.state(strategy::SkillCapability::DribbleTouch) !=
            strategy::CapabilityState::Experimental) {
        std::cerr << "enabled target kick capability was not experimental\n";
        return 1;
    }

    if (!throws_invalid_argument([] { parse({"ApolloCodeBase", "--player-number", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--player-number", "8"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--max-cycles", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--status-interval", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--team"}); }) ||
        !throws_invalid_argument([] {
            parse({"ApolloCodeBase", "--enable-fast-walk"});
        }) ||
        !throws_invalid_argument([] {
            parse({"ApolloCodeBase", "--enable-learned-kick"});
        }) ||
        !throws_invalid_argument([] {
            parse({
                "ApolloCodeBase", "--enable-parameterized-kick",
                "--enable-learned-kick"});
        }) ||
        !throws_invalid_argument([] {
            parse({
                "ApolloCodeBase", "--enable-parameterized-kick",
                "--enable-learned-kick", "--shadow-learned-kick",
                "--learned-kick-model", "/tmp/kick.onnx"});
        })) {
        std::cerr << "invalid runtime arguments were not rejected\n";
        return 1;
    }

    return 0;
}

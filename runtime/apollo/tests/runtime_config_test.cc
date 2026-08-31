// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/app/runtime_config.h"

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
        "--disable-pass-strategy",
    });

    if (config.team_name != "My3D" || config.player_number != 7 ||
        config.host != "localhost" || config.port != 61000 ||
        config.max_cycles != 800U || config.status_interval_cycles != 20U ||
        config.enable_pass_strategy) {
        std::cerr << "valid runtime arguments were not parsed correctly\n";
        return 1;
    }

    const app::RuntimeConfig enabled = parse({
        "ApolloCodeBase", "--disable-pass-strategy", "--enable-pass-strategy"});
    if (!enabled.enable_pass_strategy) {
        std::cerr << "pass strategy could not be re-enabled\n";
        return 1;
    }

    if (!throws_invalid_argument([] { parse({"ApolloCodeBase", "--player-number", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--player-number", "8"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--max-cycles", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--status-interval", "0"}); }) ||
        !throws_invalid_argument([] { parse({"ApolloCodeBase", "--team"}); })) {
        std::cerr << "invalid runtime arguments were not rejected\n";
        return 1;
    }

    return 0;
}

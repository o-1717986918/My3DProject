// SPDX-License-Identifier: GPL-3.0-or-later

#include "src/app/runtime_config.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

app::RuntimeConfig parse(std::vector<std::string> values) {
    std::vector<char*> args;
    args.reserve(values.size());
    for (std::string& value : values) {
        args.push_back(value.data());
    }
    return app::RuntimeConfig::from_args(
        static_cast<int>(args.size()), args.data());
}

}  // namespace

int main() {
    assert(parse({"ApolloCodeBase"}).enable_dynamic_pass);
    assert(!parse({"ApolloCodeBase", "--disable-dynamic-pass"})
                .enable_dynamic_pass);
    assert(parse({"ApolloCodeBase", "--disable-dynamic-pass",
                  "--enable-dynamic-pass"})
               .enable_dynamic_pass);
    assert(parse({"ApolloCodeBase"}).enable_goalkeeper_intercept);
    assert(parse({"ApolloCodeBase", "--enable-goalkeeper-intercept"})
               .enable_goalkeeper_intercept);
    assert(!parse({"ApolloCodeBase", "--enable-goalkeeper-intercept",
                   "--disable-goalkeeper-intercept"})
                .enable_goalkeeper_intercept);
    return 0;
}

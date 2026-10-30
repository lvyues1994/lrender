#pragma once

#include <tc/tc.hpp>

namespace lr::test {

using namespace tc;

struct TC_KERNEL SmokeAddOne {
    static constexpr char fileLocation[] = "lr_smoke_add_one";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> values;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(values.length())) {
            return;
        }
        values[i] = values[i] + 1u;
    }
};

} // namespace lr::test

#pragma once

// The visible instances the visibility feature compacts (VisibilityOutputs::visible):
// visible[0].x of them, each with its first visible triangle in visible[1 + slot].y, in
// increasing order.

// Member function of kernels with a BufferBinding<uvec4> `visible`: the slot of the
// visible instance whose triangles include visible triangle `triangle`.
#define LR_FIND_VISIBLE_SLOT                                                                                           \
    uint findSlot(uint triangle) {                                                                                     \
        uint low = 0u;                                                                                                 \
        uint high = visible[0].x;                                                                                      \
        while (high - low > 1u) {                                                                                      \
            uint middle = (low + high) / 2u;                                                                           \
            if (visible[1u + middle].y <= triangle) {                                                                  \
                low = middle;                                                                                          \
            } else {                                                                                                   \
                high = middle;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        return low;                                                                                                    \
    }

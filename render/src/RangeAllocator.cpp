#include "RangeAllocator.h"

#include <iterator>
#include <stdexcept>

namespace lr::render {

std::uint32_t RangeAllocator::allocate(std::uint32_t const count) {
    if (count == 0) {
        return 0;
    }
    taken += count;
    auto const fit = freeBySize.lower_bound({count, 0u});
    if (fit == freeBySize.end()) {
        auto const first = last;
        last += count;
        return first;
    }
    auto const [size, first] = *fit;
    eraseFree(freeByFirst.find(first));
    if (size > count) {
        insertFree(first + count, size - count);
    }
    return first;
}

void RangeAllocator::release(std::uint32_t first, std::uint32_t count) {
    if (count == 0) {
        return;
    }
    if (first + count > last or count > taken) {
        throw std::logic_error{"lr::render: a range released that was never allocated"};
    }
    taken -= count;
    auto const after = freeByFirst.find(first + count);
    if (after != freeByFirst.end()) {
        count += after->second;
        eraseFree(after);
    }
    auto const next = freeByFirst.lower_bound(first);
    if (next != freeByFirst.begin()) {
        auto const before = std::prev(next);
        if (before->first + before->second == first) {
            first = before->first;
            count += before->second;
            eraseFree(before);
        }
    }
    if (first + count == last) {
        last = first;
        return;
    }
    insertFree(first, count);
}

void RangeAllocator::clear() { reset(0); }

void RangeAllocator::reset(std::uint32_t const used) {
    freeByFirst.clear();
    freeBySize.clear();
    last = used;
    taken = used;
}

void RangeAllocator::insertFree(std::uint32_t const first, std::uint32_t const count) {
    freeByFirst.emplace(first, count);
    freeBySize.emplace(count, first);
}

void RangeAllocator::eraseFree(std::map<std::uint32_t, std::uint32_t>::iterator const range) {
    freeBySize.erase({range->second, range->first});
    freeByFirst.erase(range);
}

} // namespace lr::render

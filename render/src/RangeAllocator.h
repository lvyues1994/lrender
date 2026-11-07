#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <utility>

namespace lr::render {

// Places ranges of a table's elements: each in the smallest free range it fits (the lowest
// of those), or after the last one. Released ranges merge with free neighbours, and those
// at the end shorten the table.
struct RangeAllocator {
    // The first element of `count`; 0 without taking anything when `count` is 0.
    std::uint32_t allocate(std::uint32_t count);
    // [first, first + count), which allocate() returned.
    void release(std::uint32_t first, std::uint32_t count);
    // Forgets every range: what allocate() returns next starts at 0.
    void clear();
    // Starts over with [0, used) taken, as after compaction.
    void reset(std::uint32_t used);

    // One past the last element taken.
    std::uint32_t end() const noexcept { return last; }
    std::uint32_t used() const noexcept { return taken; }
    // Free elements before end().
    std::uint32_t wasted() const noexcept { return last - taken; }

private:
    void insertFree(std::uint32_t first, std::uint32_t count);
    void eraseFree(std::map<std::uint32_t, std::uint32_t>::iterator range);

    std::map<std::uint32_t, std::uint32_t> freeByFirst;         // first -> count
    std::set<std::pair<std::uint32_t, std::uint32_t>> freeBySize; // (count, first)
    std::uint32_t last = 0;
    std::uint32_t taken = 0;
};

} // namespace lr::render

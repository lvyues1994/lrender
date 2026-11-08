#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lr::render {

// The index each id has: in an array for ids below twice as many as it holds (and some),
// where ids handed out one after the other stay, else in a hash map.
template <class Id>
struct IdIndex {
    std::optional<std::uint32_t> find(Id const id) const {
        if (id.value < dense.size()) {
            auto const entry = dense[id.value];
            return entry != 0 ? std::optional<std::uint32_t>{entry - 1} : std::nullopt;
        }
        auto const found = sparse.find(id);
        return found != sparse.end() ? std::optional<std::uint32_t>{found->second} : std::nullopt;
    }

    std::uint32_t at(Id const id) const {
        auto const index = find(id);
        if (not index) {
            throw std::out_of_range{"lr::render: an id the index does not hold"};
        }
        return *index;
    }

    std::size_t count(Id const id) const { return find(id) ? 1 : 0; }

    void set(Id const id, std::uint32_t const index) {
        if (not find(id)) {
            ++held;
        }
        auto const reach = 2 * held + kSlack;
        if (id.value >= dense.size() and id.value < reach) {
            grow(std::min(reach, std::max<std::size_t>(2 * dense.size(), std::size_t{id.value} + 1)));
        }
        if (id.value < dense.size()) {
            dense[id.value] = index + 1;
        } else {
            sparse[id] = index;
        }
    }

    void erase(Id const id) {
        if (id.value < dense.size()) {
            held -= dense[id.value] != 0 ? 1 : 0;
            dense[id.value] = 0;
        } else {
            held -= sparse.erase(id);
        }
    }

private:
    static constexpr std::size_t kSlack = 1024;

    // The ids of the map that the array now reaches move into it.
    void grow(std::size_t const size) {
        dense.resize(size, 0);
        for (auto entry = sparse.begin(); entry != sparse.end();) {
            if (entry->first.value < size) {
                dense[entry->first.value] = entry->second + 1;
                entry = sparse.erase(entry);
            } else {
                ++entry;
            }
        }
    }

    std::vector<std::uint32_t> dense; // index + 1, 0 for none
    std::unordered_map<Id, std::uint32_t> sparse;
    std::size_t held = 0;
};

} // namespace lr::render

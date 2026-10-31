#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace lr::scene {

// Elements of one kind under the ids the render world knows them by, with what changed
// since the last extraction: the elements to upsert and the ids to remove. An id freed by
// erase() is handed out again only after the next extraction, so no delta upserts and
// removes the same id. Elements may carry references; release() erases at the last one.
template <class Id, class Data>
struct TrackedTable {
    Id insert(Data data) {
        auto const id = takeId();
        auto &slot = slots[id.value];
        slot.data = std::move(data);
        slot.references = 1;
        ++live;
        markChanged(id);
        return id;
    }

    bool contains(Id const id) const noexcept { return id.value < slots.size() and slots[id.value].data.has_value(); }

    Data const &operator[](Id const id) const noexcept {
        assert(contains(id));
        return *slots[id.value].data;
    }

    // The element, to be upserted again.
    Data &modify(Id const id) {
        assert(contains(id));
        markChanged(id);
        return *slots[id.value].data;
    }

    void markChanged(Id const id) {
        auto &slot = slots[id.value];
        if (not slot.changed) {
            changed.push_back(id);
            slot.changed = true;
        }
    }

    void retain(Id const id) noexcept {
        assert(contains(id));
        ++slots[id.value].references;
    }

    // Whether that was the last reference, which erased the element.
    bool release(Id const id) {
        assert(contains(id) and slots[id.value].references > 0);
        if (--slots[id.value].references != 0) {
            return false;
        }
        erase(id);
        return true;
    }

    void erase(Id const id) {
        auto &slot = slots[id.value];
        assert(slot.data.has_value());
        slot.data.reset();
        --live;
        released.push_back(id);
        if (slot.known) {
            removed.push_back(id);
            slot.known = false;
        }
    }

    std::size_t size() const noexcept { return live; }

    // Calls upsert(id, data) for each element changed since the last extraction, in the
    // order they first changed, and appends the ids removed since to `removals`.
    template <class Upsert>
    void extract(Upsert const &upsert, std::vector<Id> &removals) {
        for (auto const id : changed) {
            auto &slot = slots[id.value];
            slot.changed = false;
            if (slot.data) {
                slot.known = true;
                upsert(id, *slot.data);
            }
        }
        changed.clear();
        removals.insert(removals.end(), removed.begin(), removed.end());
        removed.clear();
        free.insert(free.end(), released.begin(), released.end());
        released.clear();
    }

private:
    struct Slot {
        std::optional<Data> data;
        std::uint32_t references = 0;
        bool known = false;   // the render world has it
        bool changed = false; // in `changed`
    };

    Id takeId() {
        if (not free.empty()) {
            auto const id = free.back();
            free.pop_back();
            return id;
        }
        slots.emplace_back();
        return Id{static_cast<std::uint32_t>(slots.size() - 1)};
    }

    std::vector<Slot> slots;
    std::vector<Id> changed;
    std::vector<Id> removed;
    std::vector<Id> released; // free after the next extraction
    std::vector<Id> free;
    std::size_t live = 0;
};

} // namespace lr::scene

#pragma once

#include <lr/scene/ScriptHost.h>
#include <lr/scene/World.h>

#include <lexec/execution.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <utility>

namespace lr::test {

using namespace std::chrono_literals;

// A value some thread provides once: senders from wait() complete with it on the thread
// that calls fire(), or at once when they start after it.
struct Trigger {
    struct State {
        std::mutex mutex;
        std::optional<int> value;
        void (*complete)(void *waiter, int value) noexcept = nullptr;
        void *waiter = nullptr;
    };

    struct Sender {
        using sender_concept = lexec::sender_t;
        using completion_signatures = lexec::completion_signatures<lexec::set_value_t(int)>;

        template <class Rcvr>
        struct Operation {
            using operation_state_concept = lexec::operation_state_t;

            void start() & noexcept {
                auto value = std::optional<int>{};
                {
                    auto const lock = std::lock_guard<std::mutex>{state->mutex};
                    value = state->value;
                    if (not value) {
                        state->complete = &Operation::complete;
                        state->waiter = this;
                    }
                }
                if (value) {
                    lexec::set_value(std::move(rcvr), *value);
                }
            }

            static void complete(void *const self, int const value) noexcept {
                lexec::set_value(std::move(static_cast<Operation *>(self)->rcvr), value);
            }

            std::shared_ptr<State> state;
            Rcvr rcvr;
        };

        template <class Rcvr>
        Operation<Rcvr> connect(Rcvr rcvr) const {
            return Operation<Rcvr>{state, std::move(rcvr)};
        }

        std::shared_ptr<State> state;
    };

    Sender wait() const { return Sender{state}; }

    void fire(int const value) const {
        auto complete = static_cast<void (*)(void *, int) noexcept>(nullptr);
        auto waiter = static_cast<void *>(nullptr);
        {
            auto const lock = std::lock_guard<std::mutex>{state->mutex};
            state->value = value;
            complete = std::exchange(state->complete, nullptr);
            waiter = std::exchange(state->waiter, nullptr);
        }
        if (complete != nullptr) {
            complete(waiter, value);
        }
    }

    std::shared_ptr<State> state = std::make_shared<State>();
};

// Completes with set_stopped once stop is requested of it, and never otherwise; counts
// the operations stopped in *stopped.
struct NeverCompletes {
    using sender_concept = lexec::sender_t;
    using completion_signatures = lexec::completion_signatures<lexec::set_stopped_t()>;

    template <class Rcvr>
    struct Operation {
        using operation_state_concept = lexec::operation_state_t;

        struct OnStop {
            void operator()() noexcept {
                op->stopped->fetch_add(1);
                lexec::set_stopped(std::move(op->rcvr));
            }

            Operation *op;
        };

        using Callback = lexec::stop_callback_for_t<lexec::stop_token_of_t<lexec::env_of_t<Rcvr>>, OnStop>;

        Operation(std::atomic<int> *const stopped_, Rcvr rcvr_) noexcept : stopped(stopped_), rcvr(std::move(rcvr_)) {}
        Operation(Operation &&) = delete;

        ~Operation() {
            if (engaged) {
                std::launder(reinterpret_cast<Callback *>(storage))->~Callback();
            }
        }

        // Once registered, the callback may complete and so destroy the operation: nothing
        // touches it after the construction.
        void start() & noexcept {
            engaged = true;
            ::new (static_cast<void *>(storage)) Callback(lexec::get_stop_token(lexec::get_env(rcvr)), OnStop{this});
        }

        std::atomic<int> *stopped;
        Rcvr rcvr;
        bool engaged = false;
        alignas(Callback) unsigned char storage[sizeof(Callback)];
    };

    template <class Rcvr>
    Operation<Rcvr> connect(Rcvr rcvr) const noexcept {
        return Operation<Rcvr>{stopped, std::move(rcvr)};
    }

    std::atomic<int> *stopped;
};

// `value` from a pool thread after `delay`, whether or not stop is requested meanwhile.
template <class Scheduler>
auto delayed(Scheduler const scheduler, std::chrono::microseconds const delay, int const value) {
    return lexec::schedule(scheduler) | lexec::then([delay, value] {
               std::this_thread::sleep_for(delay);
               return value;
           });
}

inline scene::SimDuration milliseconds(std::int64_t const count) { return std::chrono::milliseconds{count}; }

} // namespace lr::test

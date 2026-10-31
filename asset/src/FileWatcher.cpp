#include <lr/asset/FileWatcher.h>

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>
#include <stackfull/exec/io.hpp>
#include <stackfull/exec/spawn.hpp>
#include <stackfull/exec/sync.hpp>
#include <stackfull/exec/wait.hpp>
#include <stackfull/io/Registration.h>
#include <stackfull/sync/Channel.h>

#include <cerrno>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <sys/inotify.h>
#include <unistd.h>

namespace lr::asset {

namespace {

constexpr auto kWatchedEvents =
    std::uint32_t{IN_CREATE | IN_CLOSE_WRITE | IN_MOVED_TO | IN_DELETE | IN_MOVED_FROM | IN_ONLYDIR};
// Room for many events at once; each is an inotify_event and its name.
constexpr auto kReadBytes = std::size_t{64} * 1024;

using ChangeQueue = stackfull::sync::Channel<FileChange>;

struct Inotify {
    Inotify() : fd(::inotify_init1(IN_NONBLOCK | IN_CLOEXEC)) {
        if (fd < 0) {
            throw std::system_error{errno, std::generic_category(), "lr::asset: inotify_init1"};
        }
    }
    Inotify(Inotify const &) = delete;
    Inotify &operator=(Inotify const &) = delete;
    ~Inotify() { ::close(fd); }

    int fd;
};

// Which directory each inotify watch descriptor stands for.
struct WatchedDirectories {
    std::error_code add(int const inotify, std::string directory) {
        while (directory.size() > 1 and directory.back() == '/') {
            directory.pop_back();
        }
        // Held across inotify_add_watch, so the reader finds the directory of its first event.
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto const watch = ::inotify_add_watch(inotify, directory.c_str(), kWatchedEvents);
        if (watch < 0) {
            return std::error_code{errno, std::generic_category()};
        }
        directories[watch] = std::move(directory);
        return {};
    }

    std::optional<std::string> pathOf(int const watch, std::string const &name) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto const found = directories.find(watch);
        if (found == directories.end()) {
            return std::nullopt;
        }
        auto const &directory = found->second;
        return directory.back() == '/' ? directory + name : directory + "/" + name;
    }

    void forget(int const watch) {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        directories.erase(watch);
    }

private:
    std::mutex mutex;
    std::unordered_map<int, std::string> directories;
};

std::optional<FileChangeKind> kindOf(std::uint32_t const mask) noexcept {
    if ((mask & IN_Q_OVERFLOW) != 0) {
        return FileChangeKind::Overflow;
    }
    if ((mask & IN_CREATE) != 0) {
        return FileChangeKind::Created;
    }
    if ((mask & (IN_CLOSE_WRITE | IN_MOVED_TO)) != 0) {
        return FileChangeKind::Modified;
    }
    if ((mask & (IN_DELETE | IN_MOVED_FROM)) != 0) {
        return FileChangeKind::Removed;
    }
    return std::nullopt;
}

std::optional<FileChange> changeOf(inotify_event const &event, std::string const &name, WatchedDirectories &directories) {
    if ((event.mask & IN_IGNORED) != 0) {
        directories.forget(event.wd);
        return std::nullopt;
    }
    auto const kind = kindOf(event.mask);
    if (not kind) {
        return std::nullopt;
    }
    if (*kind == FileChangeKind::Overflow) {
        return FileChange{*kind, {}};
    }
    auto path = directories.pathOf(event.wd, name);
    if (not path) {
        return std::nullopt;
    }
    return FileChange{*kind, std::move(*path)};
}

// In the watcher's task: false once the watcher is stopping.
bool send(ChangeQueue &changes, FileChange change) {
    auto const sent = stackfull::exec::wait(stackfull::exec::async_send(changes, std::move(change)));
    return sent.has_value() and std::get<0>(*sent);
}

// The watcher's IO task: parked in the read until the kernel has events, and in the
// send while the queue is full, until the watcher asks it to stop.
void forwardEvents(stackfull::io::Registration &events, WatchedDirectories &directories, ChangeQueue &changes) {
    auto buffer = std::vector<char>(kReadBytes);
    for (;;) {
        auto const got = stackfull::exec::wait(stackfull::exec::async_read(events, buffer.data(), buffer.size()));
        if (not got) {
            return;
        }
        auto const bytes = std::get<0>(*got);
        auto offset = std::size_t{0};
        while (offset + sizeof(inotify_event) <= bytes) {
            auto event = inotify_event{};
            std::memcpy(&event, buffer.data() + offset, sizeof event);
            auto const *const name = buffer.data() + offset + sizeof event;
            offset += sizeof event + event.len;
            auto change = changeOf(event, std::string{name, ::strnlen(name, event.len)}, directories);
            if (change and not send(changes, std::move(*change))) {
                return;
            }
        }
    }
}

std::unique_ptr<stackfull::io::Registration> registered(stackfull::io::Poller &poller, int const fd) {
    auto registration = std::make_unique<stackfull::io::Registration>(poller, fd);
    if (registration->error()) {
        throw std::system_error{registration->error(), "lr::asset: cannot register inotify with the poller"};
    }
    return registration;
}

runtime::Runtime &runtimeOf(FileWatcherConfig const &config) {
    if (config.runtime == nullptr or config.queueCapacity == 0) {
        throw std::invalid_argument{"lr::asset: a file watcher needs a runtime and room for changes"};
    }
    return *config.runtime;
}

struct FileWatcherImpl final : FileWatcher {
    explicit FileWatcherImpl(FileWatcherConfig const &config)
        : runtime(runtimeOf(config)), events(registered(runtime.poller(), inotify.fd)), changes(config.queueCapacity) {
        lexec::spawn(stackfull::exec::spawn(runtime.io(), [this] { forwardEvents(*events, directories, changes); })
                         | lexec::upon_error([](auto const &) noexcept {}),
                     scope.get_token());
    }

    ~FileWatcherImpl() override {
        scope.request_stop();
        static_cast<void>(lexec::sync_wait(scope.join()));
    }

    std::error_code watchDirectory(std::string const &directory) override {
        return directories.add(inotify.fd, directory);
    }

    void drainChanges(std::vector<FileChange> &drained) override {
        drained.clear();
        auto change = FileChange{};
        while (changes.tryRecv(change)) {
            drained.push_back(std::move(change));
        }
    }

private:
    runtime::Runtime &runtime;
    // Destroyed bottom-up: the task is joined first, then the registration goes before
    // the descriptor closes.
    Inotify inotify;
    std::unique_ptr<stackfull::io::Registration> events;
    WatchedDirectories directories;
    ChangeQueue changes;
    lexec::counting_scope scope;
};

} // namespace

std::unique_ptr<FileWatcher> makeFileWatcher(FileWatcherConfig const &config) {
    return std::make_unique<FileWatcherImpl>(config);
}

} // namespace lr::asset

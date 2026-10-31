#pragma once

#include <lr/runtime/Runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace lr::asset {

enum class FileChangeKind : std::uint8_t {
    Created,  // content may still be on its way: a Modified follows once it is written
    Modified, // a writer closed it, or it was moved into place: its content is complete
    Removed,  // deleted or moved away
    Overflow, // the kernel dropped events: any watched file may have changed
};

struct FileChange {
    FileChangeKind kind = FileChangeKind::Modified;
    std::string path; // the watched directory, '/', the file's name; empty for Overflow
};

struct FileWatcherConfig {
    runtime::Runtime *runtime = nullptr; // required; outlives the watcher
    std::size_t queueCapacity = 1024;    // changes held for drainChanges() before events wait in the kernel
};

// Reports changes to the files of watched directories (not their subdirectories). An
// inotify descriptor registered with the runtime's poller is read by one IO task, which
// parks while nothing changes; no thread of its own.
struct FileWatcher {
    virtual ~FileWatcher() = default;

    // Any thread. Watching a directory again changes nothing.
    [[nodiscard]] virtual std::error_code watchDirectory(std::string const &directory) = 0;
    // Never waits: replaces `changes` with those reported since the last call, in order.
    virtual void drainChanges(std::vector<FileChange> &changes) = 0;
};

// Throws std::system_error when inotify or the poller cannot take another descriptor.
// Destroy it on a plain thread, before the runtime.
std::unique_ptr<FileWatcher> makeFileWatcher(FileWatcherConfig const &config);

} // namespace lr::asset

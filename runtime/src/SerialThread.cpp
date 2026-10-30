#include <lr/runtime/SerialThread.h>

namespace lr::runtime {

SerialThread::SerialThread() : thread([this] { loop.run(); }) {}

SerialThread::~SerialThread() {
    loop.finish();
    thread.join();
}

} // namespace lr::runtime

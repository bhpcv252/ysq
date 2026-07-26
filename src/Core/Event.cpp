#include <Core/Event.hpp>

#include <algorithm>

namespace ysq {

namespace {

void sweep(detail::SlotList& list) {
    std::erase_if(list.slots, [](const detail::Slot& slot) { return !slot.alive; });
    list.dirty = false;
}

/// Keeps the dispatch depth balanced even if a handler throws, and sweeps the
/// slots that were marked dead during the dispatch it closes.
class DispatchGuard {
public:
    explicit DispatchGuard(detail::SlotList& list) : m_list(list) {
        ++m_list.dispatchDepth;
    }

    DispatchGuard(const DispatchGuard&) = delete;
    DispatchGuard& operator=(const DispatchGuard&) = delete;

    ~DispatchGuard() {
        --m_list.dispatchDepth;
        if (m_list.dispatchDepth == 0 && m_list.dirty) {
            sweep(m_list);
        }
    }

private:
    detail::SlotList& m_list;
};

}  // namespace

Subscription::~Subscription() {
    unsubscribe();
}

Subscription::Subscription(Subscription&& other) noexcept
    : m_token(std::move(other.m_token)),
      m_type(other.m_type),
      m_id(other.m_id) {
    other.m_type = nullptr;
    other.m_id = 0;
}

Subscription& Subscription::operator=(Subscription&& other) noexcept {
    if (this != &other) {
        unsubscribe();
        m_token = std::move(other.m_token);
        m_type = other.m_type;
        m_id = other.m_id;
        other.m_type = nullptr;
        other.m_id = 0;
    }
    return *this;
}

void Subscription::unsubscribe() noexcept {
    if (m_type == nullptr) {
        return;
    }
    if (const std::shared_ptr<detail::BusToken> token = m_token.lock()) {
        if (token->bus != nullptr) {
            token->bus->remove(*m_type, m_id);
        }
    }
    release();
}

void Subscription::release() noexcept {
    m_token.reset();
    m_type = nullptr;
    m_id = 0;
}

EventBus::EventBus() : m_token(std::make_shared<detail::BusToken>()) {
    m_token->bus = this;
}

EventBus::~EventBus() {
    m_token->bus = nullptr;
}

void EventBus::dispatchQueued() {
    std::vector<std::function<void()>> pending;
    pending.swap(m_queue);
    for (const std::function<void()>& deliver : pending) {
        deliver();
    }
}

void EventBus::clear() {
    m_slots.clear();
    m_queue.clear();
}

void EventBus::remove(const std::type_info& type, std::uint64_t id) noexcept {
    const auto entry = m_slots.find(std::type_index{type});
    if (entry == m_slots.end()) {
        return;
    }

    detail::SlotList& list = entry->second;
    for (detail::Slot& slot : list.slots) {
        if (slot.id != id || !slot.alive) {
            continue;
        }
        // Tombstone rather than erase. This runs from a noexcept destructor, and
        // vector::erase move-assigns std::function, which is not noexcept.
        // Clearing the handler still releases its captures immediately;
        // assigning nullptr to a std::function is noexcept. The slot itself goes
        // at the next sweep, in slotsFor() or when a dispatch unwinds.
        slot.alive = false;
        slot.invoke = nullptr;
        list.dirty = true;
        return;
    }
}

detail::SlotList& EventBus::slotsFor(const std::type_info& type) {
    detail::SlotList& list = m_slots[std::type_index{type}];
    if (list.dirty && list.dispatchDepth == 0) {
        sweep(list);
    }
    return list;
}

void EventBus::dispatch(const std::type_info& type, const void* event) {
    const auto entry = m_slots.find(std::type_index{type});
    if (entry == m_slots.end()) {
        return;
    }

    // Holding a reference into the map is safe across a subscribe from inside a
    // handler: rehashing an unordered_map invalidates iterators, not references
    // to elements.
    detail::SlotList& list = entry->second;
    const DispatchGuard guard(list);

    // Snapshotting the count is what keeps a handler subscribed during this
    // dispatch from receiving the event that created it.
    const std::size_t count = list.slots.size();
    for (std::size_t i = 0; i < count; ++i) {
        if (list.slots[i].alive) {
            list.slots[i].invoke(event);
        }
    }
}

std::size_t EventBus::countAlive(const std::type_info& type) const {
    const auto entry = m_slots.find(std::type_index{type});
    if (entry == m_slots.end()) {
        return 0;
    }
    return static_cast<std::size_t>(std::count_if(
        entry->second.slots.begin(), entry->second.slots.end(),
        [](const detail::Slot& slot) { return slot.alive; }));
}

}  // namespace ysq

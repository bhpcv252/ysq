#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ysq {

class EventBus;

namespace detail {

/// Cleared when the bus dies, so a Subscription that outlives its bus becomes
/// inert instead of writing through a dangling pointer.
struct BusToken {
    EventBus* bus = nullptr;
};

struct Slot {
    std::uint64_t id = 0;
    std::function<void(const void*)> invoke;
    bool alive = true;
};

/// Handlers for one event type. Removal during dispatch is deferred: the slot is
/// marked dead and swept once the outermost dispatch unwinds, so a handler can
/// unsubscribe itself or another without disturbing the loop that is running.
struct SlotList {
    std::vector<Slot> slots;
    int dispatchDepth = 0;
    bool dirty = false;
};

}  // namespace detail

/// Move-only handle to one subscription. Unsubscribes on destruction unless
/// released, so a handler cannot outlive the object it captured by accident.
class Subscription {
public:
    Subscription() noexcept = default;
    ~Subscription();

    Subscription(const Subscription&) = delete;
    Subscription& operator=(const Subscription&) = delete;
    Subscription(Subscription&& other) noexcept;
    Subscription& operator=(Subscription&& other) noexcept;

    void unsubscribe() noexcept;

    /// Detach without unsubscribing. The handler then lives as long as the bus.
    void release() noexcept;

    [[nodiscard]] bool active() const noexcept { return m_type != nullptr; }

private:
    friend class EventBus;

    Subscription(std::weak_ptr<detail::BusToken> token, const std::type_info* type,
                 std::uint64_t id) noexcept
        : m_token(std::move(token)), m_type(type), m_id(id) {}

    std::weak_ptr<detail::BusToken> m_token;
    const std::type_info* m_type = nullptr;
    std::uint64_t m_id = 0;
};

/// Synchronous event dispatch keyed by event type.
///
/// Any copyable type is an event; there is no base class to inherit and no
/// enum to extend. Handlers may subscribe, unsubscribe and publish from inside
/// a dispatch. A handler added during a dispatch does not receive the event it
/// was created by.
///
/// Each publish costs a hash lookup on the event type plus one indirect call per
/// handler. That suits frame- and step-level events. It does not suit per-body
/// or per-sample traffic: a hot loop should call into its collaborator directly
/// and leave the bus for events something else genuinely needs to observe.
///
/// Not thread-safe. The simulation loop owns the bus.
class EventBus {
public:
    EventBus();
    ~EventBus();

    // Subscriptions refer back to this bus, so it neither copies nor moves.
    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;
    EventBus(EventBus&&) = delete;
    EventBus& operator=(EventBus&&) = delete;

    template <class E>
    [[nodiscard]] Subscription subscribe(std::function<void(const E&)> handler);

    /// Delivers now, in subscription order.
    template <class E>
    void publish(const E& event);

    /// Delivers on the next dispatchQueued().
    template <class E>
    void enqueue(E event);

    /// Delivers everything queued so far, in order. Events enqueued by a handler
    /// during this call land in the next drain rather than extending this one.
    void dispatchQueued();

    template <class E>
    [[nodiscard]] std::size_t subscriberCount() const;

    [[nodiscard]] std::size_t queuedCount() const noexcept { return m_queue.size(); }

    /// Drops every handler and every queued event. Not valid from inside a
    /// handler: it destroys the list the dispatch is walking.
    void clear();

private:
    friend class Subscription;

    void remove(const std::type_info& type, std::uint64_t id) noexcept;
    void dispatch(const std::type_info& type, const void* event);
    [[nodiscard]] std::size_t countAlive(const std::type_info& type) const;
    /// The slot list for `type`, with any tombstones reclaimed first.
    detail::SlotList& slotsFor(const std::type_info& type);

    std::shared_ptr<detail::BusToken> m_token;
    std::uint64_t m_nextId = 1;
    std::unordered_map<std::type_index, detail::SlotList> m_slots;
    std::vector<std::function<void()>> m_queue;
};

template <class E>
Subscription EventBus::subscribe(std::function<void(const E&)> handler) {
    const std::uint64_t id = m_nextId++;
    detail::SlotList& list = slotsFor(typeid(E));
    list.slots.push_back(detail::Slot{id,
                                      [callback = std::move(handler)](const void* event) {
                                          callback(*static_cast<const E*>(event));
                                      },
                                      true});
    return Subscription{m_token, &typeid(E), id};
}

template <class E>
void EventBus::publish(const E& event) {
    dispatch(typeid(E), &event);
}

template <class E>
void EventBus::enqueue(E event) {
    m_queue.push_back([this, stored = std::move(event)] { publish(stored); });
}

template <class E>
std::size_t EventBus::subscriberCount() const {
    return countAlive(typeid(E));
}

}  // namespace ysq

#include <Core/Event.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

struct Tick {
    int index = 0;
};

struct Collision {
    std::string label;
};

TEST(CoreEvent, DeliversToEverySubscriberInOrder) {
    ysq::EventBus bus;
    std::vector<std::string> calls;

    const ysq::Subscription first = bus.subscribe<Tick>(
        [&calls](const Tick& e) { calls.push_back("first:" + std::to_string(e.index)); });
    const ysq::Subscription second = bus.subscribe<Tick>([&calls](const Tick& e) {
        calls.push_back("second:" + std::to_string(e.index));
    });

    bus.publish(Tick{7});

    EXPECT_EQ(bus.subscriberCount<Tick>(), 2u);
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0], "first:7");
    EXPECT_EQ(calls[1], "second:7");
}

TEST(CoreEvent, EventTypesAreIsolated) {
    ysq::EventBus bus;
    int ticks = 0;
    int collisions = 0;

    const ysq::Subscription onTick =
        bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
    const ysq::Subscription onCollision =
        bus.subscribe<Collision>([&collisions](const Collision&) { ++collisions; });

    bus.publish(Tick{1});
    bus.publish(Tick{2});
    bus.publish(Collision{"a"});

    EXPECT_EQ(ticks, 2);
    EXPECT_EQ(collisions, 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);
}

TEST(CoreEvent, PublishingWithNoSubscribersIsHarmless) {
    ysq::EventBus bus;
    bus.publish(Tick{1});
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
}

TEST(CoreEvent, DestroyingTheSubscriptionUnsubscribes) {
    ysq::EventBus bus;
    int ticks = 0;

    {
        const ysq::Subscription subscription =
            bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
        bus.publish(Tick{1});
        EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);
    }

    bus.publish(Tick{2});
    EXPECT_EQ(ticks, 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
}

TEST(CoreEvent, ReleasedSubscriptionOutlivesItsHandle) {
    ysq::EventBus bus;
    int ticks = 0;

    {
        ysq::Subscription subscription =
            bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
        subscription.release();
        EXPECT_FALSE(subscription.active());
    }

    bus.publish(Tick{1});
    EXPECT_EQ(ticks, 1);
}

TEST(CoreEvent, MovedFromSubscriptionDoesNotUnsubscribeTwice) {
    ysq::EventBus bus;
    int ticks = 0;

    ysq::Subscription original = bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
    {
        const ysq::Subscription moved = std::move(original);
        EXPECT_FALSE(original.active());
        EXPECT_TRUE(moved.active());
        bus.publish(Tick{1});
    }

    bus.publish(Tick{2});
    EXPECT_EQ(ticks, 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);

    original.unsubscribe();  // already released; must be a no-op
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
}

TEST(CoreEvent, MoveAssignmentUnsubscribesTheOverwrittenHandler) {
    ysq::EventBus bus;
    int overwritten = 0;
    int kept = 0;

    ysq::Subscription handle =
        bus.subscribe<Tick>([&overwritten](const Tick&) { ++overwritten; });
    handle = bus.subscribe<Tick>([&kept](const Tick&) { ++kept; });

    bus.publish(Tick{1});

    EXPECT_EQ(overwritten, 0);
    EXPECT_EQ(kept, 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);
}

TEST(CoreEvent, SubscriptionOutlivingTheBusIsInert) {
    ysq::Subscription subscription;
    {
        ysq::EventBus bus;
        subscription = bus.subscribe<Tick>([](const Tick&) {});
    }
    EXPECT_TRUE(subscription.active());
    subscription.unsubscribe();  // the bus is gone; must not write through it
    EXPECT_FALSE(subscription.active());
}

TEST(CoreEvent, HandlerSubscribedDuringDispatchDoesNotSeeThatEvent) {
    ysq::EventBus bus;
    int late = 0;
    ysq::Subscription lateHandle;

    const ysq::Subscription first = bus.subscribe<Tick>([&](const Tick&) {
        if (!lateHandle.active()) {
            lateHandle = bus.subscribe<Tick>([&late](const Tick&) { ++late; });
        }
    });

    bus.publish(Tick{1});
    EXPECT_EQ(late, 0) << "a handler received the event that created it";

    bus.publish(Tick{2});
    EXPECT_EQ(late, 1);
}

TEST(CoreEvent, SubscribingToAnotherTypeDuringDispatchIsSafe) {
    ysq::EventBus bus;
    std::vector<ysq::Subscription> held;
    int ticks = 0;

    const ysq::Subscription first = bus.subscribe<Tick>([&](const Tick&) {
        ++ticks;
        // Enough new keys to force the slot map to rehash mid-dispatch.
        for (int i = 0; i < 32; ++i) {
            held.push_back(bus.subscribe<Collision>([](const Collision&) {}));
        }
    });
    const ysq::Subscription second =
        bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });

    bus.publish(Tick{1});

    EXPECT_EQ(ticks, 2);
    EXPECT_EQ(bus.subscriberCount<Collision>(), 32u);
}

TEST(CoreEvent, HandlerCanUnsubscribeAnotherDuringDispatch) {
    ysq::EventBus bus;
    int second = 0;
    ysq::Subscription secondHandle;

    const ysq::Subscription firstHandle =
        bus.subscribe<Tick>([&](const Tick&) { secondHandle.unsubscribe(); });
    secondHandle = bus.subscribe<Tick>([&second](const Tick&) { ++second; });

    bus.publish(Tick{1});
    EXPECT_EQ(second, 0) << "a handler removed during dispatch still ran";
    EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);

    bus.publish(Tick{2});
    EXPECT_EQ(second, 0);
}

TEST(CoreEvent, HandlerCanUnsubscribeItself) {
    ysq::EventBus bus;
    int calls = 0;
    ysq::Subscription handle;

    handle = bus.subscribe<Tick>([&](const Tick&) {
        ++calls;
        handle.unsubscribe();
    });

    bus.publish(Tick{1});
    bus.publish(Tick{2});

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
}

TEST(CoreEvent, NestedPublishIsSafe) {
    ysq::EventBus bus;
    std::vector<std::string> calls;

    const ysq::Subscription onTick = bus.subscribe<Tick>([&](const Tick& e) {
        calls.push_back("tick");
        if (e.index == 0) {
            bus.publish(Tick{1});  // re-enter the same slot list
        }
    });
    const ysq::Subscription onCollision = bus.subscribe<Collision>(
        [&calls](const Collision& e) { calls.push_back(e.label); });

    bus.publish(Tick{0});

    EXPECT_EQ(calls, (std::vector<std::string>{"tick", "tick"}));
    EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);
}

TEST(CoreEvent, QueuedEventsDeliverOnDispatchInOrder) {
    ysq::EventBus bus;
    std::vector<std::string> calls;

    const ysq::Subscription onTick = bus.subscribe<Tick>(
        [&calls](const Tick& e) { calls.push_back("tick:" + std::to_string(e.index)); });
    const ysq::Subscription onCollision = bus.subscribe<Collision>(
        [&calls](const Collision& e) { calls.push_back(e.label); });

    bus.enqueue(Tick{1});
    bus.enqueue(Collision{"hit"});
    bus.enqueue(Tick{2});

    EXPECT_EQ(bus.queuedCount(), 3u);
    EXPECT_TRUE(calls.empty()) << "queued events were delivered early";

    bus.dispatchQueued();

    EXPECT_EQ(calls, (std::vector<std::string>{"tick:1", "hit", "tick:2"}));
    EXPECT_EQ(bus.queuedCount(), 0u);
}

TEST(CoreEvent, EnqueuingFromAHandlerLandsInTheNextDrain) {
    ysq::EventBus bus;
    int ticks = 0;

    const ysq::Subscription onTick = bus.subscribe<Tick>([&](const Tick& e) {
        ++ticks;
        if (e.index < 3) {
            bus.enqueue(Tick{e.index + 1});
        }
    });

    bus.enqueue(Tick{1});
    bus.dispatchQueued();
    EXPECT_EQ(ticks, 1) << "the drain ran past the events it started with";
    EXPECT_EQ(bus.queuedCount(), 1u);

    bus.dispatchQueued();
    EXPECT_EQ(ticks, 2);
}

TEST(CoreEvent, ClearDropsHandlersAndQueuedEvents) {
    ysq::EventBus bus;
    int ticks = 0;

    const ysq::Subscription onTick =
        bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
    bus.enqueue(Tick{1});

    bus.clear();

    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
    EXPECT_EQ(bus.queuedCount(), 0u);
    bus.dispatchQueued();
    bus.publish(Tick{2});
    EXPECT_EQ(ticks, 0);
}

TEST(CoreEvent, HandlerCapturesSurviveByValue) {
    ysq::EventBus bus;
    auto counter = std::make_shared<int>(0);

    const ysq::Subscription onTick =
        bus.subscribe<Tick>([counter](const Tick&) { ++*counter; });
    EXPECT_EQ(counter.use_count(), 2);

    bus.publish(Tick{1});
    EXPECT_EQ(*counter, 1);
}

// Removal tombstones the slot rather than erasing it, so the handler has to be
// cleared explicitly or a captured resource would outlive its subscription.
TEST(CoreEvent, UnsubscribingReleasesTheHandlerCapturesAtOnce) {
    ysq::EventBus bus;
    auto resource = std::make_shared<int>(0);

    {
        const ysq::Subscription handle =
            bus.subscribe<Tick>([resource](const Tick&) { ++*resource; });
        EXPECT_EQ(resource.use_count(), 2);
    }

    EXPECT_EQ(resource.use_count(), 1) << "the captures outlived the subscription";
}

TEST(CoreEvent, UnsubscribingDuringDispatchStillReleasesCaptures) {
    ysq::EventBus bus;
    auto resource = std::make_shared<int>(0);
    ysq::Subscription victim = bus.subscribe<Tick>([resource](const Tick&) {});
    ASSERT_EQ(resource.use_count(), 2);

    const ysq::Subscription remover =
        bus.subscribe<Tick>([&victim](const Tick&) { victim.unsubscribe(); });
    bus.publish(Tick{1});

    EXPECT_EQ(resource.use_count(), 1);
    EXPECT_EQ(bus.subscriberCount<Tick>(), 1u);
}

// Churn against the tombstone-and-sweep path. The slot vector's length is not
// observable through the public API, so this pins the behaviour that is: each
// iteration sees exactly one live handler, and none of the dead ones fire.
TEST(CoreEvent, RepeatedSubscribeAndUnsubscribeKeepsOneLiveHandler) {
    ysq::EventBus bus;
    int ticks = 0;

    for (int i = 0; i < 1000; ++i) {
        const ysq::Subscription handle =
            bus.subscribe<Tick>([&ticks](const Tick&) { ++ticks; });
        bus.publish(Tick{i});
    }

    EXPECT_EQ(ticks, 1000) << "exactly one live handler on each iteration";
    EXPECT_EQ(bus.subscriberCount<Tick>(), 0u);
}

}  // namespace

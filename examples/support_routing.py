"""Route synthetic support tickets into in-memory queues and record each action."""
from decision_metrics import ChoiceRequest, DecisionError, Harness, JsonlSink

from _example_support import configure, print_results


TICKETS = [
    {"id": "T-001", "message": "I was charged twice for my subscription."},
    {"id": "T-002", "message": "The CSV export button returns an error."},
    {"id": "T-003", "message": "I cannot sign in after changing my email."},
]
OPTIONS = {
    "billing": {"handles": ["invoices", "payments", "refunds"]},
    "technical": {"handles": ["product errors", "exports", "integrations"]},
    "account_access": {"handles": ["sign-in problems", "passwords", "account access"]},
}


def offline_choice(request):
    message = request.state["ticket"]["message"].lower()
    if "charged" in message:
        return "billing"
    if "export" in message:
        return "technical"
    return "account_access"


def main():
    args, backend = configure(__doc__, offline_choice)
    queues = {name: [] for name in OPTIONS}
    results = []
    failures = 0
    try:
        with JsonlSink(args.output) as sink:
            bench = Harness(backend, sink, metadata={
                "application": "support_routing_example", "synthetic_inputs": True,
                "offline_demo": args.backend is None, "execution": "in_memory_queues",
            })
            for ticket in TICKETS:
                request = ChoiceRequest(
                    state={"ticket": ticket}, options=OPTIONS,
                    instructions="Choose the queue best suited to handling this support ticket.",
                )
                try:
                    decision = bench.choose(request, phase="routing", observation_id=ticket["id"],
                                            deadline_ms=args.deadline_ms)
                except DecisionError as error:
                    failures += 1
                    results.append({"ticket_id": ticket["id"], "error": error.code})
                    continue
                if decision.expired():
                    bench.record_action(decision, "expired", reason="routing_deadline")
                    results.append({"ticket_id": ticket["id"], "status": "expired"})
                    continue
                queue = decision.result.choice
                queues[queue].append(ticket["id"])
                bench.record_action(decision, "applied",
                                    details={"ticket_id": ticket["id"], "queue": queue})
                results.append({"ticket_id": ticket["id"], "queue": queue, "status": "applied"})
            bench.outcome("example_complete", metrics={
                "tickets_routed": sum(len(tickets) for tickets in queues.values()),
                "queues": queues, "failed_decisions": failures,
            })
    finally:
        backend.close()
    print_results(args.output, results)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

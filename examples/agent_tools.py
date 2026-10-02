"""Choose and execute local example tools, recording the selection and tool result."""
from decision_metrics import ChoiceRequest, DecisionError, Harness, JsonlSink

from _example_support import configure, print_results


TASKS = [
    {"id": "A-001", "question": "How do I export a CSV report?"},
    {"id": "A-002", "question": "Is the reporting service down right now?"},
    {"id": "A-003", "question": "It does not work. Can you help?"},
]
OPTIONS = {
    "search_docs": {"purpose": "Find documented instructions for using a feature"},
    "check_service_status": {"purpose": "Look up whether a service is currently available"},
    "ask_for_details": {"purpose": "Request more information when the task is unclear"},
}


def search_docs():
    return {"source": "example_documentation", "answer": "Open Reports, then select Export CSV."}


def check_service_status():
    return {"source": "example_status_fixture", "service": "reporting", "status": "operational"}


def ask_for_details():
    return {"question": "Which feature are you using, and what error do you see?"}


TOOLS = {"search_docs": search_docs, "check_service_status": check_service_status,
         "ask_for_details": ask_for_details}


def offline_choice(request):
    question = request.state["task"]["question"].lower()
    if question.startswith("how"):
        return "search_docs"
    if "down" in question:
        return "check_service_status"
    return "ask_for_details"


def main():
    args, backend = configure(__doc__, offline_choice)
    results = []
    failures = 0
    tools_executed = 0
    try:
        with JsonlSink(args.output) as sink:
            bench = Harness(backend, sink, metadata={
                "application": "agent_tool_selection_example", "synthetic_inputs": True,
                "offline_demo": args.backend is None, "execution": "local_fixture_tools",
            })
            for task in TASKS:
                request = ChoiceRequest(
                    state={"task": task}, options=OPTIONS,
                    instructions="Choose the next tool best suited to helping with this task.",
                )
                try:
                    decision = bench.choose(request, phase="tool_selection", observation_id=task["id"],
                                            deadline_ms=args.deadline_ms)
                except DecisionError as error:
                    failures += 1
                    results.append({"task_id": task["id"], "error": error.code})
                    continue
                if decision.expired():
                    bench.record_action(decision, "expired", reason="tool_selection_deadline")
                    results.append({"task_id": task["id"], "status": "expired"})
                    continue
                tool = decision.result.choice
                tool_result = TOOLS[tool]()
                tools_executed += 1
                bench.record_action(decision, "applied", details={"tool": tool, "result": tool_result})
                results.append({"task_id": task["id"], "tool": tool,
                                "result": tool_result, "status": "applied"})
            bench.outcome("example_complete", metrics={
                "tools_executed": tools_executed, "failed_decisions": failures,
            })
    finally:
        backend.close()
    print_results(args.output, results)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

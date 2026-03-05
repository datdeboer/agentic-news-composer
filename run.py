"""CLI runner for headless end-to-end testing (M3 verification)."""
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from graph.graph import get_compiled_graph
from langgraph.types import Command

STYLES = ["opinion", "newsletter", "deep-dive"]


def run(auto_approve: bool = True):
    """Run the full pipeline. If auto_approve=True, approve all drafts automatically."""
    graph, checkpointer = get_compiled_graph()
    thread_id = "cli-run"
    config = {"configurable": {"thread_id": thread_id}}

    print("Starting News Composer pipeline...")
    print("=" * 60)

    # Stream until interrupt or end
    for event in graph.stream({}, config=config, stream_mode="updates"):
        for node, output in event.items():
            print(f"\n[{node}] State updated.")

    snap = graph.get_state(config)

    # Handle human review interrupt
    while snap.next:
        interrupts = snap.tasks[0].interrupts if snap.tasks else []
        if not interrupts:
            break

        interrupt_val = interrupts[0].value
        drafts = interrupt_val.get("drafts", [])

        print("\n" + "=" * 60)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 60)

        for draft in drafts:
            print(f"\n--- {draft['style'].upper()} DRAFT ---")
            print(f"Title: {draft['title']}")
            print(draft["content"][:400] + "...\n")

        if auto_approve:
            feedback = [{"style": d["style"], "action": "approve", "notes": ""} for d in drafts]
            print("\nAuto-approving all drafts...")
        else:
            feedback = []
            for draft in drafts:
                style = draft["style"]
                choice = input(f"Approve '{style}' draft? [y/n]: ").strip().lower()
                if choice == "y":
                    feedback.append({"style": style, "action": "approve", "notes": ""})
                else:
                    notes = input(f"Revision notes for '{style}': ").strip()
                    feedback.append({"style": style, "action": "revise", "notes": notes})

        # Resume graph
        for event in graph.stream(Command(resume=feedback), config=config, stream_mode="updates"):
            for node, output in event.items():
                print(f"[{node}] State updated.")

        snap = graph.get_state(config)

    final = snap.values if snap else {}
    if final.get("finalized"):
        print("\n✓ Pipeline complete. Output written to output/<date>.md")
    else:
        print("\n Pipeline did not finalize. Check logs above.")

    return final


if __name__ == "__main__":
    auto = "--interactive" not in sys.argv
    run(auto_approve=auto)

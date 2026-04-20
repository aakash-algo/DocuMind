"""Small CLI entrypoint for quick local testing without Streamlit."""

from langchain_core.messages import HumanMessage

from agent import agent


def main():
    """Run a single question through the agent and print the trace summary."""
    question = input("Ask something: ").strip()
    if not question:
        print("Please enter a question.")
        return

    result = agent.invoke({"messages": [HumanMessage(content=question)], "llm_calls": 0})
    print("\nAnswer:\n")
    print(result.get("final_answer", "No answer generated."))

    route = result.get("route")
    if route:
        print(f"\nRoute: {route}")

    if result.get("retrieval_query"):
        print(f"Retrieval query: {result['retrieval_query']}")

    if result.get("retrieval_grade"):
        print(f"Retrieval grade: {result['retrieval_grade']}")


if __name__ == "__main__":
    main()

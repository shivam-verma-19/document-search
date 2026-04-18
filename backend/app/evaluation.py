def precision_at_k(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(retrieved)

def recall_at_k(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(relevant)

def faithfulness_score(question, answer, context, llm):
    prompt = f"""
    Check if the answer is supported by context.

    Context: {context}
    Answer: {answer}

    Score from 0 to 1:
    """
    return llm.invoke(prompt)
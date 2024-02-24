from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

class Evaluation:

    def get_dataset(self, questions, ground_truths, rag_chain, retriever):
        answers = [rag_chain.run(query) for query in questions]
        contexts = [[docs.page_content for docs in retriever.get_relevant_documents(query)] for query in questions]
        
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
        }

        return Dataset.from_dict(data)

    def evaluate_RAG(self, dataset):
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )
        return result

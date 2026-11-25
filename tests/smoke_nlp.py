"""Quick manual smoke test for the NLP layer (no DB, no API)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.nlp.evaluation import evaluate, non_reference_stats  # noqa: E402
from app.nlp.extractive import summarize  # noqa: E402
from app.nlp.keywords import extract_keywords  # noqa: E402
from app.nlp.preprocessing import preprocess  # noqa: E402
from app.nlp.quiz_generator import generate_quiz  # noqa: E402
from app.nlp.topic_modeling import generate_topics  # noqa: E402

SAMPLE = """
Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that allows computers to learn
patterns from data without being explicitly programmed for every rule. A model is
trained on a dataset and then evaluated on unseen examples.

Supervised Learning

Supervised learning uses labelled data. In classification the target is a discrete
class label, such as spam or not-spam. In regression the target is a continuous
number, such as the price of a house. Support Vector Machines (SVM) are a popular
classification algorithm that finds a maximum-margin hyperplane.

Unsupervised Learning

Unsupervised learning works with unlabelled data. Clustering groups similar examples
together, while dimensionality reduction compresses features into a smaller space.
Principal Component Analysis (PCA) is a linear dimensionality reduction technique.

Evaluation

Accuracy measures the proportion of correct predictions. Precision is the fraction of
positive predictions that are actually correct. Recall measures how many real positives
were found. The F1 score is the harmonic mean of precision and recall. A confusion
matrix summarises the counts of true positives, false positives, true negatives and
false negatives. Cross-validation is used to estimate generalisation performance.
"""

doc = preprocess(SAMPLE)
print(f"sentences={doc.sentence_count} words={doc.word_count} dupes={doc.duplicates_removed}")

res = summarize(doc.sentences, "medium")
print("\n--- TEXTRANK ---")
print(res.summary_text[:700])
print("selected:", len(res.selected), "of", res.source_sentence_count,
      "compression:", round(res.compression_ratio, 3))

kw = extract_keywords(doc.sentences, doc.clean_text, top_n=12)
print("\n--- KEYWORDS ---")
print([(k.keyword, k.score, k.frequency) for k in kw.keywords])

topics = generate_topics(doc.clean_text, doc.sentences, max_topics=4)
print("\n--- TOPICS ---")
print("method:", topics.method, "count:", len(topics.topics))
for t in topics.topics:
    print(f"  [{t.source}] {t.title} ({t.sentence_count} sents) -> {t.summary[:120]}...")

quiz = generate_quiz(doc.sentences, kw.keywords, count=5)
print("\n--- QUIZ ---")
print("generated:", quiz.generated, "skipped:", quiz.skipped, "strategies:", quiz.strategies)
for q in quiz.items:
    print("  Q:", q.question[:110])
    print("     options:", q.options, "correct:", q.correct_index)

ref = ("Machine learning learns patterns from data. Supervised learning uses labelled "
       "data for classification and regression. Unsupervised learning clusters unlabelled "
       "data. Accuracy, precision, recall and F1 score evaluate models.")
ev = evaluate(res.summary_text, ref, SAMPLE, res.processing_time, "textrank")
print("\n--- ROUGE ---")
print(ev.as_dict())
print("\n--- NON-REFERENCE ---")
print(non_reference_stats(res.summary_text, SAMPLE))

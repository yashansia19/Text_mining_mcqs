import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from transformers import pipeline

# 🔹 Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# ---------- Step 1: User chooses file ----------
file_path = input("Enter the path of your text file: ")

with open(file_path, "r", encoding="utf-8") as f:
    input_text = f.read()

print("\n✅ File loaded successfully!\n")

# ---------- Step 2: Keyword extraction ----------
r = Rake(stopwords=stopwords.words('english'))
r.extract_keywords_from_text(input_text)
keywords = r.get_ranked_phrases()[:10]  # top 10 keywords
print("🔑 Keywords found:\n", keywords)

# ---------- Step 3: Summarization (for flashcards) ----------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(input_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

print("\n📝 Summary:\n", summary)

# ---------- Step 4: Flashcard / MCQ generation ----------
def generate_flashcards(text, keywords):
    flashcards = []
    for kw in keywords:
        q = f"What is {kw}?"
        a = f"{kw} is related to: {text[:100]}..."
        flashcards.append((q, a))
    return flashcards

flashcards = generate_flashcards(input_text, keywords)

print("\n📚 Flashcards:")
for i, (q, a) in enumerate(flashcards, 1):
    print(f"\nQ{i}: {q}\nA{i}: {a}")

# ---------- Step 5: MCQ Generation (simple demo) ----------
def generate_mcq(text, keywords):
    mcqs = []
    for kw in keywords[:5]:
        question = f"Which of the following is mentioned in the text?"
        options = [kw, "Random Term 1", "Random Term 2", "Random Term 3"]
        mcqs.append((question, options, kw))
    return mcqs

mcqs = generate_mcq(input_text, keywords)

print("\n🎯 MCQs:")
for i, (q, opts, ans) in enumerate(mcqs, 1):
    print(f"\nQ{i}: {q}")
    for j, opt in enumerate(opts, 1):
        print(f"   {j}. {opt}")
    print(f"✅ Answer: {ans}")

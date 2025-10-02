from collections import Counter
import re
import math

# Read and preprocess training data
with open("train.txt", "r", encoding="utf-8") as file:
    reviews = [line.strip().lower() for line in file]

raw_tokens = []
for review in reviews:
    # Split into sentences using punctuation
    sentences = re.split(r"[.!?]+", review)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Clean text - keep only letters, digits, spaces, apostrophes
        cleaned_sentence = re.sub(r"[^a-z0-9\s']", "", sentence)
        words = cleaned_sentence.split()
        if words:
            # Add sentence boundaries for bigram context
            raw_tokens.extend(["<s>"] + words + ["</s>"])

# Count initial unigrams
initial_unigram_counts = Counter(raw_tokens)
print("Total distinct words (before filtering):", len(initial_unigram_counts))

def find_problem_words(unigram_counts):
    """Identify rare and low-quality words to replace with <UNK>"""
    single_occurrence = [w for w, c in unigram_counts.items() if c == 1]
    
    # Define quality filters
    def is_short(w): return len(w) <= 2
    def has_digits(w): return re.search(r"\d", w) is not None
    def no_vowels(w): return re.search(r"[aeiou]", w) is None
    def has_special(w): return re.search(r"[^a-z0-9\s]", w) is not None
    
    problem_words = set()
    for w in single_occurrence:
        if is_short(w) or has_digits(w) or no_vowels(w) or has_special(w):
            problem_words.add(w)
    
    print(f"Words appearing once: {len(single_occurrence)} "
          f"({len(single_occurrence)/len(unigram_counts)*100:.1f}%)")
    print(f"Problem words for <UNK>: {len(problem_words)} "
          f"({len(problem_words)/len(unigram_counts)*100:.1f}% of vocab)")
    return problem_words, single_occurrence

problem_words, single_occurrence_words = find_problem_words(initial_unigram_counts)


# Show problem words (limited sample if too many)
print("\n Problem words mapped to <UNK>")
max_show = 200
problem_list = sorted(problem_words)
if len(problem_list) <= max_show:
    print(problem_list)
else:
    print(f"First {max_show} of {len(problem_list)} problem words:")
    print(problem_list[:max_show])

# Replace problem words with <UNK>
tokens = [w if w not in problem_words else "<UNK>" for w in raw_tokens]

# Recount after UNK replacement
unigram_counts = Counter(tokens)
bigram_counts = Counter(zip(tokens, tokens[1:]))


#compute bigram probability
def bigram_prob_mle(w1, w2):
    """Unsmoothed bigram probability P(w2|w1)"""
    return bigram_counts[(w1, w2)] / unigram_counts[w1] if (w1, w2) in bigram_counts else 0.0

print("\nExample bigram probabilities (no smoothing):")
examples = [("the", "students"), ("students", "like"), ("like", "the")]
for w1, w2 in examples:
    print(f"P({w2}|{w1}) = {bigram_prob_mle(w1, w2):.4f}")

# Build final vocabulary
vocab = set(unigram_counts.keys())
V = len(vocab)
print(f"Total distinct words (after UNK mapping): {V}")

def bigram_prob(w1, w2, k=1.0):
    """Calculate smoothed bigram probability P(w2 | w1)"""
    return (bigram_counts.get((w1, w2), 0) + k) / (unigram_counts.get(w1, 0) + k * V)

def perplexity(tokens, k=1.0):
    """Compute bigram perplexity with Add-k smoothing"""
    bigrams = list(zip(tokens, tokens[1:]))
    log_sum = 0.0
    for w1, w2 in bigrams:
        p = bigram_prob(w1, w2, k)
        log_sum += math.log2(p)
    N = len(bigrams)
    return round(2 ** (-log_sum / N), 6)

# Training set perplexity
for k in [0.1, 0.5, 1.0]:
    print(f"Bigram perplexity on training set (k={k}): {perplexity(tokens, k=k)}")

# Process validation set
with open("val.txt", "r", encoding="utf-8") as file:
    val_reviews = [line.strip().lower() for line in file]

test_tokens = []
unk_count = 0
for review in val_reviews:
    sentences = re.split(r"[.!?]+", review)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        clean = re.sub(r"[^a-z0-9\s']", "", sent)
        words = clean.split()
        mapped = []
        for w in words:
            if w in vocab:
                mapped.append(w)
            else:
                mapped.append("<UNK>")
                unk_count += 1
        test_tokens.extend(["<s>"] + mapped + ["</s>"])

print(f"<UNK> tokens in test set: {unk_count}")

# Test set perplexity
for k in [0.1, 0.5, 1.0]:
    print(f"Bigram perplexity on test set (k={k}): {perplexity(test_tokens, k=k)}")
from collections import Counter
import re
import math

def preprocess_text(text):
    """Preprocess text by splitting into sentences and cleaning"""
    sentences = re.split(r"[.!?]+", text)
    words = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            # Remove unwanted characters, keep only alphanumeric, spaces and apostrophes
            clean_sent = re.sub(r"[^a-z0-9\s']", "", sent)
            words.extend(clean_sent.split())
    return words

def calculate_perplexity(token_list, prob_function, smoothing_k):
    """Calculate perplexity for a given token list using probability function"""
    log_prob_sum = sum(math.log2(prob_function(w, smoothing_k)) for w in token_list)
    return round(2 ** (-log_prob_sum / len(token_list)), 6)

# Read training data
with open("train.txt", "r", encoding="utf-8") as file:
    training_reviews = [line.strip().lower() for line in file]

# Preprocess and tokenize training data
training_tokens = []
for review in training_reviews:
    training_tokens.extend(preprocess_text(review))

# Count unigrams and build vocabulary
word_counts = Counter(training_tokens)
vocabulary = set(word_counts.keys()) | {"<UNK>"}  # Add UNK token to vocabulary
vocab_size = len(vocabulary)
total_words = sum(word_counts.values())

print("Total distinct words:", len(word_counts))

# Unigram probability function with add-k smoothing
def get_word_probability(word, smoothing_param=1.0):
    """Calculate smoothed unigram probability P(word)"""
    numerator = word_counts.get(word, 0) + smoothing_param
    denominator = total_words + smoothing_param * vocab_size
    return numerator / denominator

# Calculate perplexity on training set with different smoothing parameters
smoothing_values = [0.1, 0.5, 1.0]
for k_val in smoothing_values:
    pp = calculate_perplexity(training_tokens, get_word_probability, k_val)
    print(f"Unigram perplexity on training set (k={k_val}): {pp}")

# Read validation data
with open("val.txt", "r", encoding="utf-8") as file:
    validation_reviews = [line.strip().lower() for line in file]

# Preprocess validation data with UNK replacement
validation_tokens = []
for review in validation_reviews:
    words = preprocess_text(review)
    # Replace out-of-vocabulary words with <UNK>
    mapped_words = [w if w in vocabulary else "<UNK>" for w in words]
    validation_tokens.extend(mapped_words)

# Calculate perplexity on validation set with different smoothing parameters
for k_val in smoothing_values:
    pp = calculate_perplexity(validation_tokens, get_word_probability, k_val)
    print(f"Unigram perplexity on test set (k={k_val}): {pp}")

# Display frequency statistics
#print("\nTop 30 most frequent tokens:")
#for word, count in word_counts.most_common(30):
    #print(f"{word}: {count}")

#print("Total token count (without <s> </s>):", total_words)


#calculate the unigram probability
print("Total words in training set:", total_words)
print("Example unigram probabilities (no smoothing):")
for word in ["the", "like", "students"]:
    prob = word_counts[word] / total_words
    print(f"P({word}) = {prob:.4f}")
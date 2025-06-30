import classla
import re
import csv
from sklearn.metrics import classification_report, f1_score

# Initialize Classla pipeline
nlp = classla.Pipeline('mk', processors='tokenize,pos,lemma', use_gpu=False)

# Load a lexicon file into a set
def load_lexicon(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())

# Load all lexicons
def load_lexicons():
    return {
        'positive': load_lexicon('positive_merged.txt'),
        'negative': load_lexicon('negative_merged.txt'),
        'intensifiers': load_lexicon('MK_AnAwords_intensifiers.txt'),
        'diminishers': load_lexicon('MK_AnAwords_diminishers.txt'),
        'polarity_shifters': load_lexicon('МК_AnAwords_polarityShifters.txt'),
        'stopwords': load_lexicon('MK_AnAwords_stopwords.txt')
    }

# Normalize character repetitions
def normalize_repetitions(token):
    token = re.sub(r'(.)\1{2,}', r'\1', token)
    if len(set(token)) == 1:
        return token[0]
    return token

# Preprocess text: clean, normalize, remove stopwords, lemmatize
def preprocess(text, stopwords):
    # Remove URLs and usernames
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)

    # Tokenize and clean punctuation
    raw_tokens = [w.strip('.,!?;:„“"').lower() for w in text.split()]
    raw_tokens = [normalize_repetitions(t) for t in raw_tokens]

    # Remove stopwords
    filtered_tokens = [t for t in raw_tokens if t and t not in stopwords]

    # Lemmatize
    doc = nlp(" ".join(filtered_tokens))
    lemmatized = []
    for sentence in doc.sentences:
        for word in sentence.words:
            lemma = word.lemma.lower().strip('.,!?;:„“"')
            if lemma:
                lemmatized.append(lemma)

    return lemmatized

# Analyze sentiment score of a sentence
def analyze_sentiment(text, lexicons, epsilon=0.3):
    pos_words = lexicons['positive']
    neg_words = lexicons['negative']
    intensifiers = lexicons['intensifiers']
    diminishers = lexicons['diminishers']
    polarity_shifters = lexicons['polarity_shifters']
    stopwords = lexicons['stopwords']

    tokens = preprocess(text, stopwords)

    score = 0
    sentiment_word_count = 0

    for i in range(len(tokens)):
        word = tokens[i]
        multiplier = 1

        # Check previous word for intensifier/diminisher
        if i > 0:
            prev = tokens[i - 1]
            if prev in intensifiers:
                multiplier = 2
            elif prev in diminishers:
                multiplier = 0.5

        # Negation check: sliding window of 3 before
        negated = any(t in polarity_shifters for t in tokens[max(0, i - 3):i])

        # Determine polarity
        if word in pos_words:
            val = 1
        elif word in neg_words:
            val = -1
        else:
            continue

        sentiment_word_count += 1
        if negated:
            val *= -1
        val *= multiplier
        score += val

    normalized_score = score / sentiment_word_count if sentiment_word_count > 0 else 0

    if normalized_score > epsilon:
        return 'positive', normalized_score
    elif normalized_score < -epsilon:
        return 'negative', normalized_score
    else:
        return 'neutral', normalized_score

# Main evaluation loop
def main():
    global lexicons
    lexicons = load_lexicons()
    epsilons = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

    test_data = []

    with open('output.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            url = row[0].strip()
            sentiment_str = row[1].strip().lower()  # keep as string, lowercase
            text = ','.join(row[2:]).strip()

            if sentiment_str not in {'negative', 'neutral', 'positive'}:
                continue  # skip if label unknown

            test_data.append((text, sentiment_str))

    for epsilon in epsilons:
        y_true = []
        y_pred = []

        print(f"\n======================= Epsilon = {epsilon:.2f} =======================\n")

        for text, true_label in test_data:
            predicted_label, score = analyze_sentiment(text, lexicons, epsilon=epsilon)
            predicted_label = predicted_label.lower() if predicted_label else 'neutral'  # default fallback

            y_true.append(true_label)
            y_pred.append(predicted_label)

        print("Classification Report (Negative & Positive only):")
        print(classification_report(
            y_true, y_pred,
            digits=3,
            labels=['negative', 'positive'],
            target_names=['Negative', 'Positive'],
            zero_division=0
        ))

        f1_neg = f1_score(y_true, y_pred, labels=['negative'], average=None, zero_division=0)[0]
        f1_pos = f1_score(y_true, y_pred, labels=['positive'], average=None, zero_division=0)[0]

        avg_f1 = (f1_neg + f1_pos) / 2

        print(f"F1 Negative: {f1_neg:.3f}, F1 Positive: {f1_pos:.3f}, Average F1: {avg_f1:.3f}")

if __name__ == "__main__":
    main()
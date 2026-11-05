from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from data import LABELS, TEXTS

KEYWORD_RULES = {
    "account": ["password", "login", "account"],
    "billing": ["charged", "refund", "invoice"],
    "technical": ["crash", "error", "slow"],
}

AMBIGUOUS_TICKET = "I paid for Pro, but the dashboard still shows the free plan."


def keyword_route(ticket):
    ticket_lower = ticket.lower()
    for category, keywords in KEYWORD_RULES.items():
        if any(keyword in ticket_lower for keyword in keywords):
            return category
    return "unknown"


def show_baseline_obstacle():
    print("Keyword baseline")
    print(f"Ticket: {AMBIGUOUS_TICKET}")
    print(f"Route: {keyword_route(AMBIGUOUS_TICKET)}")
    print("Expected: billing")


def build_model():
    X_train, X_test, y_train, y_test = train_test_split(
        TEXTS,
        LABELS,
        test_size=0.25,
        random_state=42,
        stratify=LABELS,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", ComplementNB()),
        ]
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def show_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print("\nModel evaluation")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, predictions, zero_division=0))

def interactive_demo(model):
    print("\nInteractive router")
    print("Type a support ticket, or type quit to stop.")
    while True:
        ticket = input("Ticket: ").strip()
        if ticket.lower() == "quit":
            print("Demo finished.")
            break
        if not ticket:
            print("Enter a message or type quit.")
            continue
        route = model.predict([ticket])[0]
        print(f"Predicted queue: {route}")


def main():
    show_baseline_obstacle()
    model, X_test, y_test = build_model()
    show_evaluation(model, X_test, y_test)
    ml_route = model.predict([AMBIGUOUS_TICKET])[0]
    print(f"ML route: {ml_route}")
    interactive_demo(model)


if __name__ == "__main__":
    main()
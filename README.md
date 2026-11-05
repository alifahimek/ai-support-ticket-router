# AI Support Ticket Router

A local text-classification project that routes support messages to `account`, `billing`, or `technical` queues.

## Why this project exists

A keyword router can miss messages that express the right intent without using an expected keyword. This project demonstrates that failure, then replaces the rules with a trained and evaluated machine-learning pipeline.

## Approach

- 45 balanced synthetic support tickets
- Stratified train-test split
- TF-IDF text features
- Complement Naive Bayes classifier
- Accuracy and classification-report output
- Interactive terminal predictions

## Setup on Windows

```text
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```text
python router.py
```

Type a support message at the prompt. Type `quit` to exit.

## Limitations

The dataset is small and synthetic. The reported metrics demonstrate the workflow, not production readiness. A real service would need representative labeled data, broader evaluation, monitoring, privacy controls, and a human-review path.
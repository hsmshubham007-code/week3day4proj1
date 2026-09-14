import json
import sys
from pathlib import Path

from app.prompt import classify_customer_message


# Minimum acceptable pass rate
MIN_PASS_RATE = 0.90

# Location of the golden dataset
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_dataset():
    """Load the golden evaluation dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate():
    """Run the evaluation suite."""

    dataset = load_dataset()

    total = len(dataset)
    passed = 0
    failed = 0

    print("=" * 60)
    print("EVALUATION STARTED")
    print("=" * 60)

    for case in dataset:
        case_id = case["id"]
        message = case["input"]
        expected = case["expected"]

        predicted = classify_customer_message(message)

        if predicted == expected:
            result = "PASS"
            passed += 1
        else:
            result = "FAIL"
            failed += 1

        print(f"[{result}] Case {case_id}")
        print(f"  Input:    {message}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print()

    pass_rate = passed / total if total else 0

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total cases : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Pass rate   : {pass_rate:.2%}")
    print(f"Required    : {MIN_PASS_RATE:.2%}")
    print()

    if pass_rate >= MIN_PASS_RATE:
        print("RESULT: PASS")
        print("Evaluation threshold satisfied.")
        print("=" * 60)
        return 0

    print("RESULT: FAIL")
    print("Evaluation threshold NOT satisfied.")
    print("=" * 60)

    return 1


if __name__ == "__main__":
    sys.exit(evaluate())
import csv
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

from app.prompt import classify_customer_message


# ---------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------

# Minimum acceptable absolute pass rate
MIN_PASS_RATE = 0.90

# Maximum allowed drop compared with the previous evaluation
MAX_ALLOWED_DROP = 0.05

# Prompt version
PROMPT_VERSION = "v3"


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

LOG_PATH = Path(__file__).parent / "eval_log.csv"

OBSERVABILITY_LOG_PATH = (
    Path(__file__).parent / "observability_log.csv"
)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_dataset():
    """Load the golden evaluation dataset."""

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Previous evaluation
# ---------------------------------------------------------

def get_previous_pass_rate():
    """
    Read the most recent evaluation result from the CSV log.

    Returns None if no previous evaluation exists.
    """

    if not LOG_PATH.exists():
        return None

    with open(
        LOG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        return None

    previous_row = rows[-1]

    return (
        float(
            previous_row["overall_pass_rate"]
            .replace("%", "")
        )
        / 100
    )


# ---------------------------------------------------------
# Evaluation history log
# ---------------------------------------------------------

def append_eval_log(
    total,
    passed,
    failed,
    overall_pass_rate,
    category_stats,
    result,
):
    """Append the overall evaluation results to the CSV log."""

    file_exists = LOG_PATH.exists()

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "prompt_version",
                "date",
                "total_cases",
                "passed",
                "failed",
                "overall_pass_rate",
                "billing_rate",
                "technical_rate",
                "delivery_rate",
                "account_rate",
                "result",
            ])

        def category_rate(category):

            stats = category_stats.get(
                category,
                {
                    "total": 0,
                    "passed": 0
                }
            )

            if stats["total"] == 0:
                return 0

            return (
                stats["passed"]
                / stats["total"]
            )

        writer.writerow([
            PROMPT_VERSION,
            date.today().isoformat(),
            total,
            passed,
            failed,
            f"{overall_pass_rate:.2%}",
            f"{category_rate('Billing'):.2%}",
            f"{category_rate('Technical'):.2%}",
            f"{category_rate('Delivery'):.2%}",
            f"{category_rate('Account'):.2%}",
            result,
        ])


# ---------------------------------------------------------
# Observability log
# ---------------------------------------------------------

def append_observability_log(
    timestamp,
    query_type,
    passed,
    failure_category,
    latency_ms,
):
    """
    Append one evaluation case to the observability log.

    Cost is currently 0 because this evaluator uses
    a deterministic local classifier rather than an LLM API.
    """

    file_exists = OBSERVABILITY_LOG_PATH.exists()

    with open(
        OBSERVABILITY_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "timestamp",
                "prompt_version",
                "query_type",
                "passed",
                "failure_category",
                "latency_ms",
                "cost_usd",
            ])

        writer.writerow([
            timestamp,
            PROMPT_VERSION,
            query_type,
            passed,
            failure_category,
            f"{latency_ms:.2f}",
            "0.0000",
        ])


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate():
    """Run the evaluation suite with regression detection."""

    dataset = load_dataset()

    total = len(dataset)

    passed = 0

    failed = 0

    category_stats = {}

    print("=" * 60)
    print("EVALUATION STARTED")
    print("=" * 60)

    # -----------------------------------------------------
    # Evaluate every case
    # -----------------------------------------------------

    for case in dataset:

        case_id = case["id"]

        message = case["input"]

        expected = case["expected"]

        query_type = case["query_type"]


        # -----------------------------------------------
        # Initialize category statistics
        # -----------------------------------------------

        if query_type not in category_stats:

            category_stats[query_type] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
            }

        category_stats[
            query_type
        ]["total"] += 1


        # -----------------------------------------------
        # Measure evaluation latency
        # -----------------------------------------------

        start_time = time.perf_counter()

        predicted = classify_customer_message(
            message
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        # -----------------------------------------------
        # Determine result
        # -----------------------------------------------

        timestamp = datetime.now().isoformat()

        if predicted == expected:

            result = "PASS"

            passed_case = 1

            passed += 1

            category_stats[
                query_type
            ]["passed"] += 1

            failure_category = "None"

        else:

            result = "FAIL"

            passed_case = 0

            failed += 1

            category_stats[
                query_type
            ]["failed"] += 1

            failure_category = "Wrong_Category"


        # -----------------------------------------------
        # Save observability record
        # -----------------------------------------------

        append_observability_log(
            timestamp=timestamp,
            query_type=query_type,
            passed=passed_case,
            failure_category=failure_category,
            latency_ms=latency_ms,
        )


        # -----------------------------------------------
        # Print result
        # -----------------------------------------------

        print(
            f"[{result}] Case {case_id}"
        )

        print(
            f"  Query type: {query_type}"
        )

        print(
            f"  Input:      {message}"
        )

        print(
            f"  Expected:   {expected}"
        )

        print(
            f"  Predicted:  {predicted}"
        )

        print(
            f"  Latency:    {latency_ms:.2f} ms"
        )

        print()


    # -----------------------------------------------------
    # Overall pass rate
    # -----------------------------------------------------

    pass_rate = (
        passed / total
        if total
        else 0
    )


    print("=" * 60)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Total cases : {total}"
    )

    print(
        f"Passed      : {passed}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        f"Pass rate   : {pass_rate:.2%}"
    )

    print(
        f"Required    : {MIN_PASS_RATE:.2%}"
    )


    # -----------------------------------------------------
    # Regression detection
    # -----------------------------------------------------

    previous_pass_rate = (
        get_previous_pass_rate()
    )

    regression_detected = False


    if previous_pass_rate is not None:

        drop = (
            previous_pass_rate
            - pass_rate
        )

        print()
        print("=" * 60)
        print("PROMPT REGRESSION CHECK")
        print("=" * 60)

        print(
            f"Previous pass rate : "
            f"{previous_pass_rate:.2%}"
        )

        print(
            f"Current pass rate  : "
            f"{pass_rate:.2%}"
        )

        print(
            f"Allowed drop       : "
            f"{MAX_ALLOWED_DROP:.2%}"
        )

        print(
            f"Actual drop        : "
            f"{drop:.2%}"
        )


        if drop > MAX_ALLOWED_DROP:

            regression_detected = True

            print(
                "Regression detected: YES"
            )

        else:

            print(
                "Regression detected: NO"
            )

    else:

        print()

        print(
            "No previous evaluation found."
        )

        print(
            "Regression comparison skipped."
        )


    # -----------------------------------------------------
    # Stratified results
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("STRATIFIED RESULTS BY QUERY TYPE")
    print("=" * 60)


    for query_type, stats in category_stats.items():

        category_pass_rate = (
            stats["passed"]
            / stats["total"]
            if stats["total"]
            else 0
        )

        print(
            f"\n{query_type}"
        )

        print(
            f"  Total      : "
            f"{stats['total']}"
        )

        print(
            f"  Passed     : "
            f"{stats['passed']}"
        )

        print(
            f"  Failed     : "
            f"{stats['failed']}"
        )

        print(
            f"  Pass rate  : "
            f"{category_pass_rate:.2%}"
        )


    # -----------------------------------------------------
    # Final CI decision
    # -----------------------------------------------------

    if pass_rate < MIN_PASS_RATE:

        final_result = "FAIL"

        print()

        print(
            "Failure reason: "
            "pass rate is below minimum threshold."
        )

    elif regression_detected:

        final_result = "FAIL"

        print()

        print(
            "Failure reason: "
            "pass rate dropped too much "
            "from the previous version."
        )

    else:

        final_result = "PASS"


    # -----------------------------------------------------
    # Save evaluation history
    # -----------------------------------------------------

    append_eval_log(
        total=total,
        passed=passed,
        failed=failed,
        overall_pass_rate=pass_rate,
        category_stats=category_stats,
        result=final_result,
    )


    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()

    print("=" * 60)

    print(
        f"RESULT: {final_result}"
    )

    print("=" * 60)


    return (
        0
        if final_result == "PASS"
        else 1
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    sys.exit(
        evaluate()
    )


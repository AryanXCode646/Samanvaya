from samanvaya.validation.claim_gate import evaluate_claims


def test_claim_gate_requires_independent_evidence():
    claims = evaluate_claims({"status": "SUCCESS", "registered_output": "registered_source.tif", "validation_status": "REAL_REGISTRATION_PENDING_INDEPENDENT_VALIDATION"})
    assert claims["REAL_REGISTRATION_SUPPORTED"] == "PROVEN"
    assert claims["INDEPENDENT_ACCURACY_SUPPORTED"] == "DATA_REQUIRED"
    assert claims["SUBPIXEL_ACCURACY_SUPPORTED"] == "DATA_REQUIRED"


def test_claim_gate_does_not_promote_failed_run():
    claims = evaluate_claims({"status": "NO_CORRESPONDENCE"})
    assert all(value == "DATA_REQUIRED" or value == "NOT_AVAILABLE" for key, value in claims.items() if key.endswith("SUPPORTED"))

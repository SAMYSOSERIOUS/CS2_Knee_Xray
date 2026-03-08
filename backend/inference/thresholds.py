"""
Clinical Thresholds for Decision Making
Extracted from NB3 utility analysis and cost-optimization
"""

# Optimized thresholds from NB3 clinical analysis
# These are tuned for 95% sensitivity (minimize false negatives)

CLINICAL_THRESHOLDS = {
    "Screening": 0.35,    # KL≥1: High sensitivity threshold (conservative)
    "OA": 0.45,           # KL≥2: Standard threshold
    "Severe": 0.50,       # KL≥3: Balanced threshold
}

# Performance metrics at these thresholds (source: NB3 binary results, seed 42)
THRESHOLD_METRICS = {
    "Screening": {
        "threshold": 0.35,
        "sensitivity": 0.79,
        "specificity": 0.91,
        "auc": 0.926,
        "accuracy": 0.842,
        "task": "KL≥1 (Normal vs Any OA)",
    },
    "OA": {
        "threshold": 0.45,
        "sensitivity": 0.80,
        "specificity": 0.94,
        "auc": 0.952,
        "accuracy": 0.878,
        "task": "KL≥2 (Normal/Doubtful vs Definite OA)",
    },
    "Severe": {
        "threshold": 0.50,
        "sensitivity": 0.91,
        "specificity": 0.97,
        "auc": 0.990,
        "accuracy": 0.961,
        "task": "KL≥3 (Moderate vs Severe OA)",
    },
}

# Decision thresholds for uncertainty
UNCERTAINTY_THRESHOLDS = {
    "high": 0.1,      # |prob - 0.5| < 0.1 (roughly ±5 percentage points from 50%)
    "moderate": 0.2,  # |prob - 0.5| < 0.2
    "low": 1.0,       # Default
}

# Cost-benefit analysis (from NB3 utility function)
COST_RATIOS = {
    "false_negative": 10,   # High cost of missing OA
    "false_positive": 1,    # Lower cost of conservative referral
    "explanation": "Missing progressive OA has high clinical cost (disease progression). "
                   "Over-referral has lower cost (patient education + serial imaging)."
}

# Traffic light decision rules
TRAFFIC_LIGHT_RULES = {
    "green": {
        "condition": "Stage 1 prob < 0.35",
        "label": "Low Risk / No Definite OA",
        "recommendation": "Standard preventive care",
        "followup": "Routine screening in 2-3 years",
    },
    "yellow": {
        "condition": "Stage 1 prob ≥ 0.35 AND Stage 2 prob < 0.50",
        "label": "Moderate OA (KL-2)",
        "recommendation": "Conservative treatment (PT, NSAIDs, weight management)",
        "followup": "Serial imaging in 12-24 months",
    },
    "red": {
        "condition": "Stage 1 prob ≥ 0.35 AND Stage 2 prob ≥ 0.50",
        "label": "Severe OA (KL-3/4)",
        "recommendation": "Orthopedic consultation for surgical planning",
        "followup": "Urgent orthopedic evaluation",
    },
}

# Comparison with baseline
PERFORMANCE_COMPARISON = {
    "5_class_baseline": {
        "accuracy": 0.713,
        "approach": "Traditional multi-class classification",
        "source": "NB2",
    },
    "4_class_cascade": {
        "accuracy": 0.722,
        "approach": "Hierarchical binary cascade (Screening + OA Detection + Severity)",
        "source": "NB7",
        "improvement": 0.006,  # +0.6 pp over 5-class baseline
        "improvement_pct": "0.6 pp",
    },
}


def get_threshold_for_cost_ratio(cost_fn_fp: float):
    """
    Dynamically compute threshold for custom cost ratio (FN:FP).
    
    This would typically involve computing utility function from validation data.
    For now, returns the standard threshold.
    """
    # In production, this would read from a lookup table or compute via
    # binary search on validation set for given cost ratio
    return CLINICAL_THRESHOLDS["Screening"]


def get_clinical_recommendation(traffic_light: str) -> str:
    """Get recommendation string from traffic light color."""
    return TRAFFIC_LIGHT_RULES.get(traffic_light, {}).get("recommendation", "Consult clinician")


def get_followup_timeline(traffic_light: str) -> str:
    """Get follow-up recommendation from traffic light color."""
    return TRAFFIC_LIGHT_RULES.get(traffic_light, {}).get("followup", "Routine screening")

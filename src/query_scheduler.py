"""
Query allocation scheduler for decision-based attacks.

Implements 3-stage scheduling used in adversarial attack papers:

Stage 1 — Initialization
Stage 2 — Boundary Localization
Stage 3 — Boundary Refinement
"""


def allocate_queries(max_queries, attack):

    attack = attack.lower()

    # -------------------------------------------------
    # Attack-specific query distributions
    # -------------------------------------------------
    if attack == "ba" or attack == "boundary":

        # Boundary Attack is very query heavy
        init_ratio = 0.15
        local_ratio = 0.45

    elif attack == "hsja":

        # HSJA quickly finds boundary
        init_ratio = 0.10
        local_ratio = 0.50

    elif attack == "qeba":

        # QEBA uses gradient estimation
        init_ratio = 0.15
        local_ratio = 0.45

    elif attack == "surfree":

        # SurFree benefits from stronger initialization
        init_ratio = 0.20
        local_ratio = 0.40

    elif attack == "geoda":

        # GeoDA uses geometric gradient estimation
        init_ratio = 0.15
        local_ratio = 0.35

    else:

        # default fallback
        init_ratio = 0.15
        local_ratio = 0.45

    # -------------------------------------------------
    # Compute query budgets
    # -------------------------------------------------
    init_q = int(max_queries * init_ratio)
    local_q = int(max_queries * local_ratio)
    refine_q = max_queries - init_q - local_q

    # -------------------------------------------------
    # Return cumulative query checkpoints
    # -------------------------------------------------
    return {
        "initialization": init_q,
        "localization": init_q + local_q,
        "refinement": max_queries
    }
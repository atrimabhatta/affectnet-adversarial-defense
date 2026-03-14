ATTACK_CONFIG = {

    "BA": {
        "max_queries": 25000,
        "schedule": "staged",
        "init_ratio": 0.15,
        "loc_ratio": 0.45,
        "ref_ratio": 0.40
    },

    "HSJA": {
        "max_queries": 10000,
        "schedule": "direct",
        "num_gradient": 80,
        "binary_search_steps": 10,
        "theta": 0.01
    },

    "QEBA": {
        "max_queries": 5000,
        "schedule": "direct",
        "subspace_dim": 100,
        "gradient_samples": 60
    },

    "SurFree": {
        "max_queries": 4000,
        "schedule": "direct",
        "subspace_dim": 50,
        "num_directions": 20
    },

    "GeoDA": {
        "max_queries": 3000,
        "schedule": "direct",
        "num_iterations": 30,
        "queries_per_iteration": 120
    }

}
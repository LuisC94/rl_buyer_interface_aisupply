import math
import random
import datetime
import numpy as np
import threading
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import uuid
from datetime import datetime, timedelta

# =============================================================================
# 1. FRUIT PRESETS & CLASSIFICATION
# =============================================================================

FRUIT_GROUPS = {
    "strawberry": "very_fragile",
    "raspberry": "very_fragile",
    "fig": "very_fragile",
    "blueberry": "fragile_tough_skin",
    "cherry": "fragile_tough_skin",
    "banana": "climacteric_tactile",
    "kiwi_hayward": "crunchy_texture",
    "kiwi_baby": "crunchy_texture",
    "kiwi_gold": "crunchy_texture",
    "apple_golden": "crunchy_texture",
    "apple_reineta": "crunchy_texture",
    "apple_gala": "crunchy_texture",
    "apple_fuji": "crunchy_texture",
    "apple_granny_smith": "crunchy_texture",
    "apple_red_delicious": "crunchy_texture",
    "apple_bravo_esmolfe": "crunchy_texture",
    "apple_royal_gold": "crunchy_texture",
    "apple_pink_lady": "crunchy_texture",
    "apple_jonagold": "crunchy_texture",
    "apple_alcobaca": "crunchy_texture",
    "pear": "crunchy_texture",
    "peach": "juicy_flavor",
    "plum": "juicy_flavor",
    "melon": "juicy_flavor",
    "grape": "pure_non_climacteric",
    "orange": "citrus",
}

STAKEHOLDER_CONFIG_BY_GROUP = {
    "very_fragile": {
        "Producer":  {"w_4d": (0.50, 0.20, 0.20, 0.10), "w_2d": (0.50, 0.50), "min_quality": 70.0, "mold_limit": 0.01},
        "Processor": {"w_4d": (0.75, 0.10, 0.10, 0.05), "w_2d": (0.75, 0.25), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.60, 0.20, 0.15, 0.05), "w_2d": (0.60, 0.40), "min_quality": 50.0, "mold_limit": 0.02},
        "Industry":  {"w_4d": (0.05, 0.30, 0.50, 0.15), "w_2d": (0.05, 0.95), "min_quality": 35.0, "mold_limit": 0.00},
    },
    "fragile_tough_skin": {
        "Producer":  {"w_4d": (0.45, 0.25, 0.20, 0.10), "w_2d": (0.45, 0.55), "min_quality": 70.0, "mold_limit": 0.01},
        "Processor": {"w_4d": (0.70, 0.15, 0.10, 0.05), "w_2d": (0.70, 0.30), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.55, 0.30, 0.10, 0.05), "w_2d": (0.55, 0.45), "min_quality": 50.0, "mold_limit": 0.015},
        "Industry":  {"w_4d": (0.10, 0.30, 0.45, 0.15), "w_2d": (0.10, 0.90), "min_quality": 35.0, "mold_limit": 0.01},
    },
    "climacteric_tactile": {
        "Producer":  {"w_4d": (0.65, 0.15, 0.10, 0.10), "w_2d": (0.65, 0.35), "min_quality": 70.0, "mold_limit": 0.01},
        "Processor": {"w_4d": (0.75, 0.10, 0.10, 0.05), "w_2d": (0.75, 0.25), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.45, 0.30, 0.15, 0.10), "w_2d": (0.45, 0.55), "min_quality": 50.0, "mold_limit": 0.015},
        "Industry":  {"w_4d": (0.05, 0.30, 0.50, 0.15), "w_2d": (0.05, 0.95), "min_quality": 35.0, "mold_limit": 0.02},
    },
    "crunchy_texture": {
        "Producer":  {"w_4d": (0.55, 0.15, 0.20, 0.10), "w_2d": (0.55, 0.45), "min_quality": 70.0, "mold_limit": 0.01},
        "Processor": {"w_4d": (0.65, 0.15, 0.10, 0.10), "w_2d": (0.65, 0.35), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.45, 0.30, 0.15, 0.10), "w_2d": (0.45, 0.55), "min_quality": 50.0, "mold_limit": 0.01},
        "Industry":  {"w_4d": (0.10, 0.25, 0.45, 0.20), "w_2d": (0.10, 0.90), "min_quality": 35.0, "mold_limit": 0.00},
    },
    "juicy_flavor": {
        "Producer":  {"w_4d": (0.50, 0.20, 0.20, 0.10), "w_2d": (0.50, 0.50), "min_quality": 70.0, "mold_limit": 0.02},
        "Processor": {"w_4d": (0.60, 0.15, 0.15, 0.10), "w_2d": (0.60, 0.40), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.35, 0.40, 0.15, 0.10), "w_2d": (0.35, 0.65), "min_quality": 50.0, "mold_limit": 0.01},
        "Industry":  {"w_4d": (0.05, 0.35, 0.45, 0.15), "w_2d": (0.05, 0.95), "min_quality": 35.0, "mold_limit": 0.01},
    },
    "pure_non_climacteric": {
        "Producer":  {"w_4d": (0.30, 0.30, 0.30, 0.10), "w_2d": (0.30, 0.70), "min_quality": 70.0, "mold_limit": 0.02},
        "Processor": {"w_4d": (0.50, 0.20, 0.20, 0.10), "w_2d": (0.50, 0.50), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.35, 0.40, 0.15, 0.10), "w_2d": (0.35, 0.65), "min_quality": 50.0, "mold_limit": 0.015},
        "Industry":  {"w_4d": (0.05, 0.35, 0.45, 0.15), "w_2d": (0.05, 0.95), "min_quality": 35.0, "mold_limit": 0.05},
    },
    "citrus": {
        "Producer":  {"w_4d": (0.20, 0.35, 0.30, 0.15), "w_2d": (0.20, 0.80), "min_quality": 70.0, "mold_limit": 0.01},
        "Processor": {"w_4d": (0.40, 0.30, 0.20, 0.10), "w_2d": (0.40, 0.60), "min_quality": 65.0, "mold_limit": 0.00},
        "Retailer":  {"w_4d": (0.25, 0.45, 0.20, 0.10), "w_2d": (0.25, 0.75), "min_quality": 50.0, "mold_limit": 0.01},
        "Industry":  {"w_4d": (0.05, 0.45, 0.35, 0.15), "w_2d": (0.05, 0.95), "min_quality": 35.0, "mold_limit": 0.01},
    }
}

PACKAGING_FACTORS = {
    "bulk": 1.0,
    "open_box": 0.85,
    "perforated_bag": 0.45,
    "map_sealed": 0.10
}

# Unified Physical Presets per Fruit
PRESETS = {
    "kiwi_hayward": {
        "label": "Kiwi (Hayward)", "Tref_C": 5.0, "Ea_J": 82000, "k_firm_ref": 0.024, "beta_RH": 1.15, "RH_ref": 90,
        "brix_min": 8.36, "brix_max": 15.08, "brix_g": 0.32, "brix_0_default": 11.0, "qual_brix_target": 15.0,
        "firmness_min": 5.24, "firmness_0_default": 45.0, "qual_firmness_threshold": 10.0,
        "acidity_0_default": 1.5, "acidity_min": 0.31, "k_acidity_ref": 0.013, "Ea_acidity_J": 55000, "qual_acidity_target": 1.0,
        "SL_ref": 115.0, "Ea_SL_J": 45000.0, "E0_int": 0.02, "Eref_prod": 0.12, "E_t0": 10, "E_g": 0.9, "E_auto": 0.35, "E_decay": 0.7, "Ea_E_J": 52000, "E_ext_shift": 2.0, "alpha_E": 2.0,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0, "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0,
        "senescent_firmness_collapse": 5.5
    },
    "kiwi_baby": {
        "label": "Kiwi (Baby/Berry)", "Tref_C": 4.0, "Ea_J": 60000, "k_firm_ref": 0.035, "beta_RH": 2.0, "RH_ref": 95,
        "brix_min": 8.0, "brix_max": 18.0, "brix_g": 0.5, "brix_0_default": 14.5, "qual_brix_target": 17.0,
        "firmness_min": 2.0, "firmness_0_default": 28.0, "qual_firmness_threshold": 9.0,
        "acidity_0_default": 1.1, "acidity_min": 0.5, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 1.0,
        "SL_ref": 26.0, "Ea_SL_J": 45000.0, "E0_int": 0.03, "Eref_prod": 0.22, "E_t0": 5, "E_g": 1.2, "E_auto": 0.55, "E_decay": 0.75, "Ea_E_J": 52000, "E_ext_shift": 2.6, "alpha_E": 3.2,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.07, "mold_sens_RH": 10.0, "mold_max_penalty": 0.75, "Ea_mold_J": 45000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0,
        "senescent_firmness_collapse": 5.0
    },
    "kiwi_gold": {
        "label": "Kiwi (Gold)",
        "Tref_C": 5.0,
        "Ea_J": 75000,
        "k_firm_ref": 0.020,
        "beta_RH": 1.20,
        "RH_ref": 90,
        "brix_min": 10.0,
        "brix_max": 17.5,
        "brix_g": 0.35,
        "brix_0_default": 13.0,
        "qual_brix_target": 16.0,
        "firmness_min": 4.5,
        "firmness_0_default": 40.0,
        "qual_firmness_threshold": 8.0,
        "acidity_0_default": 0.95,
        "acidity_min": 0.35,
        "k_acidity_ref": 0.015,
        "Ea_acidity_J": 55000,
        "qual_acidity_target": 0.70,
        "SL_ref": 110.0,
        "Ea_SL_J": 45000.0,
        "E0_int": 0.02,
        "Eref_prod": 0.15,
        "E_t0": 8,
        "E_g": 1.0,
        "E_auto": 0.45,
        "E_decay": 0.70,
        "Ea_E_J": 52000,
        "E_ext_shift": 2.2,
        "alpha_E": 2.0,
        "RH_mold_thr": 95.0,
        "mold_rate_ref": 0.055,
        "mold_sens_RH": 9.5,
        "mold_max_penalty": 0.70,
        "Ea_mold_J": 43000.0,
        "is_climacteric": True,
        "CI_type": "threshold",
        "CI_thr": 1.5,
        "CI_thr_min": 1.0,
        "CI_thr_max": 1.5,
        "CI_rate": 0.012,
        "CI_sl_stress": 1.5,
        "CI_max_penalty": 0.35,
        "senescent_firmness_collapse": 5.0
    },
    "apple_gala": {
        "label": "Apple (Gala)", "Tref_C": 5.0, "Ea_J": 64000, "k_firm_ref": 0.0085, "beta_RH": 0.9, "RH_ref": 90,
        "brix_min": 11.5, "brix_max": 18.5, "brix_g": 0.25, "brix_0_default": 13.0, "qual_brix_target": 14.5,
        "firmness_min": 9.0, "firmness_0_default": 60.0, "qual_firmness_threshold": 28.0,
        "acidity_0_default": 0.4, "acidity_min": 0.2, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.5,
        "SL_ref": 75.0, "Ea_SL_J": 42000.0, "E0_int": 0.015, "Eref_prod": 0.18, "E_t0": 10, "E_g": 0.9, "E_auto": 0.5, "E_decay": 0.65, "Ea_E_J": 52000, "E_ext_shift": 2.2, "alpha_E": 1.3,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0, "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "apple_golden": {
        "label": "Apple (Golden)", "Tref_C": 5.0, "Ea_J": 75000, "k_firm_ref": 0.012, "beta_RH": 0.8, "RH_ref": 90,
        "brix_min": 11.5, "brix_max": 19.0, "brix_g": 0.18, "brix_0_default": 13.5, "qual_brix_target": 14.5,
        "firmness_min": 12.0, "firmness_0_default": 72.0, "qual_firmness_threshold": 35.0,
        "acidity_0_default": 0.5, "acidity_min": 0.25, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.6,
        "SL_ref": 120.0, "Ea_SL_J": 45000.0, "E0_int": 0.01, "Eref_prod": 0.10, "E_t0": 18, "E_g": 0.6, "E_auto": 0.35, "E_decay": 0.55, "Ea_E_J": 52000, "E_ext_shift": 1.8, "alpha_E": 0.8,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.04, "mold_sens_RH": 8.0, "mold_max_penalty": 0.60, "Ea_mold_J": 42000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "apple_fuji": {
        "label": "Apple (Fuji)", "Tref_C": 5.0, "Ea_J": 85000, "k_firm_ref": 0.008, "beta_RH": 0.7, "RH_ref": 90,
        "brix_min": 13.0, "brix_max": 19.0, "brix_g": 0.15, "brix_0_default": 14.0, "qual_brix_target": 16.0,
        "firmness_min": 15.0, "firmness_0_default": 80.0, "qual_firmness_threshold": 32.0,
        "acidity_0_default": 0.4, "acidity_min": 0.2, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.5,
        "SL_ref": 180.0, "Ea_SL_J": 60000.0, "E0_int": 0.008, "Eref_prod": 0.06, "E_t0": 25, "E_g": 0.5, "E_auto": 0.25, "E_decay": 0.45, "Ea_E_J": 52000, "E_ext_shift": 1.4, "alpha_E": 0.6,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.035, "mold_sens_RH": 8.0, "mold_max_penalty": 0.55, "Ea_mold_J": 42000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "apple_reineta": {
        "label": "Apple (Reineta)", "Tref_C": 5.0, "Ea_J": 80000, "k_firm_ref": 0.018, "beta_RH": 1.0, "RH_ref": 90,
        "brix_min": 11.0, "brix_max": 16.0, "brix_g": 0.16, "brix_0_default": 11.5, "qual_brix_target": 12.5,
        "firmness_min": 10.0, "firmness_0_default": 65.0, "qual_firmness_threshold": 30.0,
        "acidity_0_default": 0.8, "acidity_min": 0.3, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.7,
        "SL_ref": 90.0, "E0_int": 0.01, "Eref_prod": 0.13, "E_t0": 14, "E_g": 0.7, "E_auto": 0.4, "E_decay": 0.6, "Ea_E_J": 52000, "E_ext_shift": 2.0, "alpha_E": 1.1,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.05, "mold_sens_RH": 9.0, "mold_max_penalty": 0.65, "Ea_mold_J": 43000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "apple_red_delicious": {
        "label": "Apple (Red Delicious)",
        "Tref_C": 5.0,
        "Ea_J": 49000,
        "k_firm_ref": 0.028,
        "beta_RH": 0.85,
        "RH_ref": 90,
        "brix_min": 11.0,
        "brix_max": 16.5,
        "brix_g": 0.20,
        "brix_0_default": 12.5,
        "qual_brix_target": 13.5,
        "firmness_min": 18.0,
        "firmness_0_default": 76.0,
        "qual_firmness_threshold": 44.0,
        "acidity_0_default": 0.30,
        "acidity_min": 0.15,
        "k_acidity_ref": 0.015,
        "Ea_acidity_J": 55000,
        "qual_acidity_target": 0.30,
        "SL_ref": 160.0,
        "E0_int": 0.012,
        "Eref_prod": 0.14,
        "E_t0": 14,
        "E_g": 0.75,
        "E_auto": 0.40,
        "E_decay": 0.60,
        "Ea_E_J": 52000,
        "E_ext_shift": 2.0,
        "alpha_E": 1.1,
        "RH_mold_thr": 95.0,
        "mold_rate_ref": 0.045,
        "mold_sens_RH": 8.5,
        "mold_max_penalty": 0.60,
        "Ea_mold_J": 42000.0,
        "is_climacteric": True,
        "CI_thr": 0.0,
        "CI_rate": 0.0,
        "CI_sl_stress": 1.0,
        "CI_max_penalty": 0.0,
    },
    "pear": {
        "label": "Pear (Rocha)", "Tref_C": 0.0, "Ea_J": 80000, "k_firm_ref": 0.0080, "beta_RH": 1.2, "RH_ref": 92,
        "brix_min": 10.5, "brix_max": 16.5, "brix_g": 0.28, "brix_0_default": 11.5, "qual_brix_target": 14.0,
        "firmness_min": 4.0, "firmness_0_default": 55.0, "qual_firmness_threshold": 20.0,
        "acidity_0_default": 0.3, "acidity_min": 0.15, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.4,
        "SL_ref": 180.0, "Ea_SL_J": 45000.0, "E0_int": 0.01, "Eref_prod": 0.22, "E_t0": 10, "E_g": 1.0, "E_auto": 0.6, "E_decay": 0.75, "Ea_E_J": 56000, "E_ext_shift": 2.3, "alpha_E": 1.6,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.07, "mold_sens_RH": 10.0, "mold_max_penalty": 0.75, "Ea_mold_J": 45000.0,
        "is_climacteric": True, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0,
        "senescent_firmness_collapse": 8.5
    },
    "strawberry": {
        "label": "Strawberry", "Tref_C": 2.0, "Ea_J": 52000, "k_firm_ref": 0.12, "beta_RH": 2.4, "RH_ref": 95,
        "brix_min": 6.0, "brix_max": 10.5, "brix_g": 0.12, "brix_0_default": 7.5, "qual_brix_target": 9.0,
        "firmness_min": 1.5, "firmness_0_default": 12.0, "qual_firmness_threshold": 4.5,
        "acidity_0_default": 0.8, "acidity_min": 0.25, "k_acidity_ref": 0.025, "Ea_acidity_J": 55000, "qual_acidity_target": 0.8,
        "SL_ref": 7.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.9, "Ea_E_J": 42000, "E_ext_shift": 0.4, "alpha_E": 0.05,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.22, "mold_sens_RH": 16.0, "mold_max_penalty": 0.95, "Ea_mold_J": 52000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "raspberry": {
        "label": "Raspberry", "Tref_C": 2.0, "Ea_J": 45000, "k_firm_ref": 0.10, "beta_RH": 2.2, "RH_ref": 95,
        "brix_min": 7.0, "brix_max": 12.0, "brix_g": 0.22, "brix_0_default": 9.5, "qual_brix_target": 10.0,
        "firmness_min": 2.5, "firmness_0_default": 18.0, "qual_firmness_threshold": 11.0,
        "acidity_0_default": 1.2, "acidity_min": 0.28, "k_acidity_ref": 0.023, "Ea_acidity_J": 55000, "qual_acidity_target": 1.0,
        "SL_ref": 7.0, "Ea_SL_J": 45000.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.9, "Ea_E_J": 42000, "E_ext_shift": 0.4, "alpha_E": 0.12,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.20, "mold_sens_RH": 16.0, "mold_max_penalty": 0.95, "Ea_mold_J": 52000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "blueberry": {
        "label": "Blueberry", "Tref_C": 2.0, "Ea_J": 45000, "k_firm_ref": 0.026, "beta_RH": 1.6, "RH_ref": 95,
        "brix_min": 10.0, "brix_max": 14.0, "brix_g": 0.2, "brix_0_default": 11.5, "qual_brix_target": 12.5,
        "firmness_min": 6.0, "firmness_0_default": 30.0, "qual_firmness_threshold": 22.0,
        "acidity_0_default": 0.6, "acidity_min": 0.3, "k_acidity_ref": 0.002, "Ea_acidity_J": 55000, "qual_acidity_target": 0.7,
        "SL_ref": 22.0, "Ea_SL_J": 35000.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.9, "Ea_E_J": 42000, "E_ext_shift": 0.4, "alpha_E": 0.10,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.12, "mold_sens_RH": 14.0, "mold_max_penalty": 0.90, "Ea_mold_J": 48000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "cherry": {
        "label": "Cherry", "Tref_C": 2.0, "Ea_J": 48000, "k_firm_ref": 0.045, "beta_RH": 1.6, "RH_ref": 95,
        "brix_min": 14.0, "brix_max": 20.0, "brix_g": 0.08, "brix_0_default": 16.0, "qual_brix_target": 18.0,
        "firmness_min": 4.0, "firmness_0_default": 28.0, "qual_firmness_threshold": 18.0,
        "acidity_0_default": 0.5, "acidity_min": 0.3, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.6,
        "SL_ref": 16.0, "Ea_SL_J": 42000.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.9, "Ea_E_J": 42000, "E_ext_shift": 0.4, "alpha_E": 0.05,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.10, "mold_sens_RH": 13.0, "mold_max_penalty": 0.85, "Ea_mold_J": 47000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "banana": {
        "label": "Banana", "Tref_C": 14.0, "Ea_J": 65000, "k_firm_ref": 0.09, "beta_RH": 1.0, "RH_ref": 90,
        "brix_min": 12.0, "brix_max": 22.0, "brix_g": 0.45, "brix_0_default": 12.5, "qual_brix_target": 19.0,
        "firmness_min": 5.0, "firmness_0_default": 80.0, "qual_firmness_threshold": 15.0,
        "acidity_0_default": 0.4, "acidity_min": 0.2, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.4,
        "SL_ref": 10.0, "E0_int": 0.02, "Eref_prod": 0.35, "E_t0": 4, "E_g": 1.4, "E_auto": 0.7, "E_decay": 0.9, "Ea_E_J": 60000, "E_ext_shift": 3.0, "alpha_E": 2.2,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.06, "mold_sens_RH": 10.0, "mold_max_penalty": 0.80, "Ea_mold_J": 45000.0,
        "is_climacteric": True, "CI_type": "threshold", "CI_thr": 13.0, "CI_thr_min": 11.5, "CI_thr_max": 13.0, "CI_rate": 0.04, "CI_sl_stress": 3.5, "CI_max_penalty": 0.95
    },
    "peach": {
        "label": "Peach", "Tref_C": 2.0, "Ea_J": 56000, "k_firm_ref": 0.08, "beta_RH": 1.1, "RH_ref": 92,
        "brix_min": 9.5, "brix_max": 18.0, "brix_g": 0.35, "brix_0_default": 11.0, "qual_brix_target": 15.0,
        "firmness_min": 2.0, "firmness_0_default": 35.0, "qual_firmness_threshold": 6.0,
        "acidity_0_default": 0.6, "acidity_min": 0.3, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.6,
        "SL_ref": 21.0, "E0_int": 0.012, "Eref_prod": 0.22, "E_t0": 6, "E_g": 1.0, "E_auto": 0.55, "E_decay": 0.8, "Ea_E_J": 56000, "E_ext_shift": 2.4, "alpha_E": 1.4,
        "RH_mold_thr": 94.0, "mold_rate_ref": 0.10, "mold_sens_RH": 12.0, "mold_max_penalty": 0.85, "Ea_mold_J": 48000.0,
        "is_climacteric": True,
        "CI_type": "zone", "CI_zone_min": 2.0, "CI_zone_max": 7.0, "CI_zone_peak": 5.0,
        "CI_thr": 7.0, "CI_rate": 0.035, "CI_sl_stress": 1.1, "CI_max_penalty": 0.65
    },
    "plum": {
        "label": "Plum", "Tref_C": 2.0, "Ea_J": 52000, "k_firm_ref": 0.040, "beta_RH": 1.0, "RH_ref": 92,
        "brix_min": 11.0, "brix_max": 20.0, "brix_g": 0.3, "brix_0_default": 12.5, "qual_brix_target": 16.5,
        "firmness_min": 3.0, "firmness_0_default": 40.0, "qual_firmness_threshold": 7.0,
        "acidity_0_default": 0.8, "acidity_min": 0.3, "k_acidity_ref": 0.016, "Ea_acidity_J": 55000, "qual_acidity_target": 0.8,
        "SL_ref": 28.0, "Ea_SL_J": 45000.0, "E0_int": 0.01, "Eref_prod": 0.14, "E_t0": 8, "E_g": 0.8, "E_auto": 0.45, "E_decay": 0.7, "Ea_E_J": 54000, "E_ext_shift": 2.0, "alpha_E": 1.0,
        "RH_mold_thr": 94.0, "mold_rate_ref": 0.09, "mold_sens_RH": 12.0, "mold_max_penalty": 0.82, "Ea_mold_J": 47000.0,
        "is_climacteric": True,
        "CI_type": "zone", "CI_zone_min": 2.0, "CI_zone_max": 7.0, "CI_zone_peak": 5.0,
        "CI_thr": 7.0, "CI_rate": 0.022, "CI_sl_stress": 1.8, "CI_max_penalty": 0.60,
        "senescent_firmness_collapse": 5.0
    },
    "fig": {
        "label": "Fig", "Tref_C": 2.0, "Ea_J": 52000, "k_firm_ref": 0.08, "beta_RH": 1.8, "RH_ref": 95,
        "brix_min": 14.0, "brix_max": 26.0, "brix_g": 0.18, "brix_0_default": 16.0, "qual_brix_target": 20.0,
        "firmness_min": 1.2, "firmness_0_default": 10.0, "qual_firmness_threshold": 5.0,
        "acidity_0_default": 0.3, "acidity_min": 0.15, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.3,
        "SL_ref": 8.0, "Ea_SL_J": 48000.0, "E0_int": 0.005, "Eref_prod": 0.05, "E_t0": 12, "E_g": 0.5, "E_auto": 0.15, "E_decay": 0.7, "Ea_E_J": 52000, "E_ext_shift": 1.0, "alpha_E": 0.2,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.18, "mold_sens_RH": 14.0, "mold_max_penalty": 0.92, "Ea_mold_J": 50000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "melon": {
        "label": "Melon", "Tref_C": 7.0, "Ea_J": 52000, "k_firm_ref": 0.06, "beta_RH": 0.9, "RH_ref": 90,
        "brix_min": 9.0, "brix_max": 16.0, "brix_g": 0.22, "brix_0_default": 10.5, "qual_brix_target": 11.5,
        "firmness_min": 2.0, "firmness_0_default": 25.0, "qual_firmness_threshold": 6.0,
        "acidity_0_default": 0.2, "acidity_min": 0.1, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.2,
        "SL_ref": 24.0, "Ea_SL_J": 45000.0, "E0_int": 0.01, "Eref_prod": 0.10, "E_t0": 10, "E_g": 0.7, "E_auto": 0.35, "E_decay": 0.7, "Ea_E_J": 52000, "E_ext_shift": 1.8, "alpha_E": 0.9,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.08, "mold_sens_RH": 10.0, "mold_max_penalty": 0.78, "Ea_mold_J": 45000.0,
        "is_climacteric": True, "CI_type": "threshold", "CI_thr": 5.0, "CI_thr_min": 2.0, "CI_thr_max": 5.0, "CI_rate": 0.045, "CI_sl_stress": 1.8, "CI_max_penalty": 0.50
    },
    "grape": {
        "label": "Grape", "Tref_C": 2.0, "Ea_J": 45000, "k_firm_ref": 0.030, "beta_RH": 1.8, "RH_ref": 92,
        "brix_min": 14.0, "brix_max": 22.0, "brix_g": 0.05, "brix_0_default": 16.0, "qual_brix_target": 18.0,
        "firmness_min": 3.0, "firmness_0_default": 20.0, "qual_firmness_threshold": 12.0,
        "acidity_0_default": 0.6, "acidity_min": 0.3, "k_acidity_ref": 0.02, "Ea_acidity_J": 55000, "qual_acidity_target": 0.6,
        "SL_ref": 35.0, "Ea_SL_J": 42000.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.9, "Ea_E_J": 42000, "E_ext_shift": 0.4, "alpha_E": 0.05,
        "RH_mold_thr": 95.0, "mold_rate_ref": 0.10, "mold_sens_RH": 14.0, "mold_max_penalty": 0.90, "Ea_mold_J": 46000.0,
        "is_climacteric": False, "CI_thr": 0.0, "CI_rate": 0.0, "CI_sl_stress": 1.0, "CI_max_penalty": 0.0
    },
    "orange": {
        "label": "Orange", "Tref_C": 5.0, "Ea_J": 40000, "k_firm_ref": 0.012, "beta_RH": 1.0, "RH_ref": 90,
        "brix_min": 10.5, "brix_max": 13.5, "brix_g": 0.1, "brix_0_default": 11.5, "qual_brix_target": 12.2,
        "firmness_min": 35.0, "firmness_0_default": 55.0, "qual_firmness_threshold": 46.0,
        "acidity_0_default": 1.0, "acidity_min": 0.3, "k_acidity_ref": 0.005, "Ea_acidity_J": 55000, "qual_acidity_target": 1.0,
        "SL_ref": 120.0, "Ea_SL_J": 42000.0, "E0_int": 0.002, "Eref_prod": 0.01, "E_t0": 999, "E_g": 0.2, "E_auto": 0.0, "E_decay": 0.8, "Ea_E_J": 40000, "E_ext_shift": 0.5, "alpha_E": 0.15,
        "RH_mold_thr": 96.0, "mold_rate_ref": 0.03, "mold_sens_RH": 8.0, "mold_max_penalty": 0.50, "Ea_mold_J": 42000.0,
        "is_climacteric": False, "CI_type": "threshold", "CI_thr": 3.0, "CI_rate": 0.012, "CI_sl_stress": 1.5, "CI_max_penalty": 0.40
    }
}

# Compatibility aliases
PRESETS_SOFIA = PRESETS
PRESETS_ACADEMIC = PRESETS

def get_fruit_group(fruit_key: str) -> str:
    return FRUIT_GROUPS.get(fruit_key, "crunchy_texture")

def get_stakeholder_config(fruit_key: str, stakeholder: str) -> dict:
    group = get_fruit_group(fruit_key)
    cfg = STAKEHOLDER_CONFIG_BY_GROUP.get(group, STAKEHOLDER_CONFIG_BY_GROUP["crunchy_texture"])
    if stakeholder in cfg:
        return cfg[stakeholder]
    for k in cfg:
        if k.lower() in stakeholder.lower() or stakeholder.lower() in k.lower():
            return cfg[k]
    return cfg.get("Retailer", list(cfg.values())[0])

# =============================================================================
# 2. METEOROLOGY & WAREHOUSE SIMULATION (IPMA)
# =============================================================================

def get_region_weather(region_code, target_date):
    profiles = {
        'PT-NL': (14.5, 5.0, 83.0, 5.0),
        'PT-NI': (13.5, 10.0, 72.0, 12.0),
        'PT-CL': (15.5, 6.0, 78.0, 6.0),
        'PT-CI': (14.0, 9.0, 70.0, 10.0),
        'PT-LVT': (17.0, 7.0, 72.0, 8.0),
        'PT-AL': (17.5, 11.0, 64.0, 15.0),
        'PT-ALG': (18.5, 6.5, 67.0, 8.0),
        'PT-SM': (8.5, 9.5, 78.0, 12.0),
        'PT-MAD': (19.5, 3.5, 74.0, 4.0),
        'PT-ACO': (17.5, 3.0, 85.0, 4.0),
    }
    base_temp, temp_amp, base_hum, hum_amp = profiles.get(region_code, (16.0, 7.0, 75.0, 8.0))
    day_of_year = target_date.timetuple().tm_yday
    seasonal_factor = math.sin(2 * math.pi * (day_of_year - 110) / 365)
    temp = round(base_temp + temp_amp * seasonal_factor, 1)
    hum = round(base_hum - hum_amp * seasonal_factor, 1)
    return temp, max(10.0, min(100.0, hum))

def calc_absolute_humidity(T_cels, RH_pct):
    numerator = 2.16679 * RH_pct * 6.112 * math.exp((17.67 * T_cels) / (T_cels + 243.5))
    denominator = T_cels + 273.15
    return numerator / denominator

def calc_relative_humidity(T_cels, HA):
    numerator = HA * (T_cels + 273.15)
    denominator = 2.16679 * 6.112 * math.exp((17.67 * T_cels) / (T_cels + 243.5))
    HR = numerator / denominator
    return min(HR, 100.0)

def simulate_warehouse_climate(T_ext_series, HR_ext_series, alphas=None, T_int_initial=None):
    if not T_ext_series or len(T_ext_series) != len(HR_ext_series):
        return [], []
    if alphas is None:
        alphas = [0.9] * len(T_ext_series)
    elif not isinstance(alphas, list):
        alphas = [alphas] * len(T_ext_series)
        
    T_int_series = []
    HR_int_series = []
    T_int_prev = T_ext_series[0] if T_int_initial is None else T_int_initial
    
    for i in range(len(T_ext_series)):
        T_ext = T_ext_series[i]
        HR_ext = HR_ext_series[i]
        alpha = alphas[i]
        T_int = alpha * T_int_prev + (1 - alpha) * T_ext
        T_int_series.append(round(T_int, 2))
        HA_ext = calc_absolute_humidity(T_ext, HR_ext)
        HR_int = calc_relative_humidity(T_int, HA_ext)
        HR_int_series.append(round(HR_int, 2))
        T_int_prev = T_int
        
    return T_int_series, HR_int_series

def fill_nulls_with_warehouse_sim(temp_array, rh_array, region_codes, alphas, start_date):
    T_ext_series = []
    HR_ext_series = []
    for i in range(len(temp_array)):
        current_date = start_date + timedelta(days=i)
        temp, hum = get_region_weather(region_codes[i] if i < len(region_codes) else 'PT-LVT', current_date)
        T_ext_series.append(temp)
        HR_ext_series.append(hum)
    T_int_series, HR_int_series = simulate_warehouse_climate(T_ext_series, HR_ext_series, alphas=alphas)
    
    res_temp = [T_int_series[i] if t is None else t for i, t in enumerate(temp_array)]
    res_rh = [HR_int_series[i] if r is None else r for i, r in enumerate(rh_array)]
    return res_temp, res_rh

# =============================================================================
# 3. MODEL 1: WITH ETHYLENE (Prof. Luís Paulo - 2D Stakeholder Quality)
# =============================================================================

def run_simulation_prof_luis_paulo(fruit_key: str, T_c: list[float], E_ext_ppm: list[float], RH_pct: list[float], 
                                   days: int, firmness_0_user: float, brix_0_user: float, 
                                   packaging_methods: Optional[list[str]] = None,
                                   current_owner_type: Optional[str] = "Retailer (Grocery Store)",
                                   dt: float = 0.05, custom_preset: dict = None):
    R = 8.314

    def k_temp_scaling(Ea, T, Tref):
        return np.exp((-Ea / R) * (1/T - 1/Tref))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    p = custom_preset if custom_preset is not None else PRESETS[fruit_key]
    is_climacteric = bool(p.get("is_climacteric", True))
    CI_type = p.get("CI_type", "threshold")
    CI_zone_min = float(p.get("CI_zone_min", 2.0))
    CI_zone_max = float(p.get("CI_zone_max", 7.0))
    CI_zone_peak = float(p.get("CI_zone_peak", 5.0))
    CI_thr = float(p.get("CI_thr", 0.0))
    CI_rate = float(p.get("CI_rate", 0.0))
    CI_sl_stress = float(p.get("CI_sl_stress", 1.0))
    CI_max_penalty = float(p.get("CI_max_penalty", 0.0))

    st_cfg = get_stakeholder_config(fruit_key, current_owner_type)
    w_firm, w_brix = st_cfg["w_2d"]
    min_quality = st_cfg["min_quality"]
    mold_limit = st_cfg["mold_limit"]

    if packaging_methods is None:
        packaging_methods = ['bulk'] * days
    else:
        packaging_methods = [pm if pm is not None else 'bulk' for pm in packaging_methods]

    # Projection Window (365 days max) repeating last known values
    projection_days = 365
    max_sim_days = days + projection_days
    t = np.arange(0, max_sim_days + dt * 0.5, dt)

    def stretch_with_last_value(arr):
        arr_rep = np.repeat(np.array(arr), int(1/dt))
        if len(arr_rep) < len(t):
            arr_rep = np.append(arr_rep, [arr_rep[-1]] * (len(t) - len(arr_rep)))
        return arr_rep[:len(t)]

    T_c_ext = stretch_with_last_value(T_c)
    RH_pct_ext = stretch_with_last_value(RH_pct)
    E_ext_ext = stretch_with_last_value(E_ext_ppm if isinstance(E_ext_ppm, list) else [E_ext_ppm]*days)
    pkg_ext = stretch_with_last_value(packaging_methods)

    T_K = T_c_ext + 273.15
    Tref_K = p["Tref_C"] + 273.15
    RH_ref = p["RH_ref"]

    # 1. Endogenous Ethylene
    E_int = np.zeros_like(t)
    E_int[0] = float(p["E0_int"])
    Ea_E_J = float(p["Ea_E_J"])
    Eref_prod = float(p["Eref_prod"])
    E_decay = float(p["E_decay"])
    E_t0 = float(p["E_t0"])
    E_g = float(p["E_g"])
    E_auto = float(p["E_auto"])
    E_ext_shift = float(p["E_ext_shift"])

    t0_eff = E_t0 - E_ext_shift * np.log1p(np.maximum(0.0, E_ext_ext))
    prod_T = k_temp_scaling(Ea_E_J, T_K, Tref_K)

    for i in range(1, len(t)):
        ramp = sigmoid(E_g * (t[i-1] - t0_eff[i-1]))
        prod = Eref_prod * prod_T[i-1] * ramp
        dE = (prod * (1.0 + E_auto * E_int[i-1]) - E_decay * E_int[i-1]) * dt
        E_int[i] = max(0.0, E_int[i-1] + dE)

    E_total = E_ext_ext + E_int

    # 2. Firmness ODE (with Packaging Factor)
    kT_firm = p["k_firm_ref"] * k_temp_scaling(p["Ea_J"], T_K, Tref_K)
    firmness = np.zeros_like(t)
    firmness_min = float(p["firmness_min"])
    firmness[0] = max(firmness_min + 1e-6, float(firmness_0_user))
    alpha_E = float(p.get("alpha_E", 0.1))

    for i in range(1, len(t)):
        pkg_method = pkg_ext[i-1]
        fator_embalagem = PACKAGING_FACTORS.get(pkg_method, 1.0)
        resp_factor = 0.5 + 0.5 * fator_embalagem

        kE = (1.0 + alpha_E * E_total[i-1])
        RH_deficit = max(0.0, (RH_ref - RH_pct_ext[i-1]) / 100.0)
        kRH = (1.0 + p["beta_RH"] * RH_deficit * fator_embalagem)
        k = kT_firm[i-1] * resp_factor * kRH * kE
        dD = (-k * (firmness[i-1] - firmness_min)) * dt
        firmness[i] = max(firmness_min, firmness[i-1] + dD)

    # 3. Brix ODE
    brix = np.zeros_like(t)
    brix_min = float(p["brix_min"])
    brix_max = float(p["brix_max"])
    brix[0] = float(brix_0_user)
    r0 = float(p.get("brix_g", 0.1))
    alpha_bE = 0.25
    rT = k_temp_scaling(Ea_E_J, T_K, Tref_K)

    for i in range(1, len(t)):
        pkg_method = pkg_ext[i-1]
        fator_embalagem = PACKAGING_FACTORS.get(pkg_method, 1.0)
        resp_factor = 0.5 + 0.5 * fator_embalagem

        if is_climacteric:
            bRH = max(0.0, (RH_ref - RH_pct_ext[i-1]) / 100.0) * fator_embalagem
            rRH = (1.0 - 0.6 * bRH)
            r = r0 * rT[i-1] * rRH * resp_factor * (1.0 + alpha_bE * E_total[i-1])
            x = max(0.01, brix[i-1] - brix_min)
            K = max(1e-6, (brix_max - brix_min))
            db = (r * x * (1.0 - x / K)) * dt
            brix[i] = min(brix_max, max(brix_min, brix[i-1] + db))
        else:
            r_resp = 0.005 * rT[i-1] * resp_factor
            db = -r_resp * dt
            brix[i] = max(brix_min, brix[i-1] + db)

    # 4. Mold Development, Cold Injury & Physical Shelf Life Cap
    RH_mold_thr = float(p["RH_mold_thr"])
    mold_rate_ref = float(p["mold_rate_ref"])
    mold_sens_RH = float(p["mold_sens_RH"])
    mold_max_penalty = float(p["mold_max_penalty"])
    Ea_mold_J = float(p["Ea_mold_J"])
    mold_T = k_temp_scaling(Ea_mold_J, T_K, Tref_K)

    SL_ref = float(p.get("SL_ref", 60.0))
    consumed_SL = np.zeros_like(t)

    mold = np.zeros_like(t)
    cold_injury = np.zeros_like(t)
    for i in range(1, len(t)):
        # Acumulação de Dano por Frio (Cold Injury)
        if CI_type == "zone":
            curr_T = T_c_ext[i-1]
            if CI_zone_min <= curr_T <= CI_zone_max:
                if curr_T <= CI_zone_peak:
                    ratio = (curr_T - CI_zone_min) / max(1e-3, (CI_zone_peak - CI_zone_min))
                else:
                    ratio = (CI_zone_max - curr_T) / max(1e-3, (CI_zone_max - CI_zone_peak))
                delta_T_cold = max(0.0, ratio) * (CI_zone_max - CI_zone_min) * 0.5
            else:
                delta_T_cold = 0.0
        else:
            delta_T_cold = max(0.0, CI_thr - T_c_ext[i-1])
        dCI = CI_rate * delta_T_cold * dt
        cold_injury[i] = min(1.0, cold_injury[i-1] + dCI)

        RH_excess = max(0.0, (RH_pct_ext[i-1] - RH_mold_thr) / 100.0)
        RH_factor = 1.0 - np.exp(-mold_sens_RH * RH_excess)
        cold_mold_inhibition = np.clip((T_c_ext[i-1] - (-1.0)) / 10.0, 0.05, 1.0)
        rate = mold_rate_ref * mold_T[i-1] * RH_factor * cold_mold_inhibition
        dm = (rate * (1.0 - mold[i-1])) * dt
        mold[i] = min(1.0, max(0.0, mold[i-1] + dm))

        pkg_method = pkg_ext[i-1]
        fator_embalagem = PACKAGING_FACTORS.get(pkg_method, 1.0)
        resp_factor = 0.5 + 0.5 * fator_embalagem
        RH_deficit = max(0.0, (RH_ref - RH_pct_ext[i-1]) / 100.0)
        r_T_SL = k_temp_scaling(float(p.get("Ea_SL_J", 45000)), T_K[i-1], Tref_K) * resp_factor
        r_RH_SL = 1.0 + 0.5 * RH_deficit * fator_embalagem
        ci_stress_factor = 1.0 + (CI_sl_stress - 1.0) * cold_injury[i-1]
        consumed_SL[i] = consumed_SL[i-1] + (r_T_SL * r_RH_SL * ci_stress_factor) * dt

    remaining_SL_fisica_cap = np.maximum(0, SL_ref - consumed_SL)

    # 5. Continuous Smooth Quality (0-100) & Biological Sanity Synchronization
    firm_score = 1.0 / (1.0 + np.exp(-0.35 * (firmness - float(p["qual_firmness_threshold"]))))
    target_b = float(p["qual_brix_target"])
    brix_score = np.where(brix >= target_b, 1.0, np.exp(-((brix - target_b)**2) / 8.0))
    group = get_fruit_group(fruit_key)
    if group in ["crunchy_texture", "climacteric_tactile", "juicy_flavor", "citrus"]:
        texture_factor = np.minimum(1.0, 2.0 * firm_score)
    else:
        texture_factor = 1.0
    quality_raw = 100.0 * (w_firm * firm_score + w_brix * brix_score) * texture_factor
    
    # Fator de Sanidade Biológica: atinge 0.0 quando o bolor atinge o limiar crítico de 50%
    sanity_factor = np.maximum(0.0, 1.0 - (mold / 0.50))
    
    # Fator de Senescência Fisiológica: atinge 0.0 quando a shelf-life física de referência é esgotada
    cap_decay = np.clip(remaining_SL_fisica_cap / max(1.0, SL_ref * 0.10), 0.0, 1.0)

    # Fator de Penalização de Frio (Cold Injury)
    ci_factor = np.maximum(0.0, 1.0 - (CI_max_penalty * cold_injury))
    
    # Qualidade Global Integrada
    quality = np.maximum(0.0, np.minimum(100.0, quality_raw * sanity_factor * cap_decay * ci_factor))

    # 6. Milestone Points Calculation
    # Point A: Commercial Life Limit (The last day the fruit meets commercial standards before permanent expiration)
    marketable = (quality >= min_quality) & (mold <= mold_limit) & (remaining_SL_fisica_cap > 0)
    ok_indices = np.where(marketable)[0]
    if len(ok_indices) > 0:
        last_ok_idx = ok_indices[-1]
        point_a_day = float(t[last_ok_idx])
    else:
        point_a_day = 0.0

    # Point B: Biological Decay / Rotting (Senescência celular, bolor crítico ou colapso interno)
    bio_decay_factor = sanity_factor * cap_decay * ci_factor
    senesc_f = float(p.get("senescent_firmness_collapse", 8.5 if fruit_key == "pear" else 0.0))
    if senesc_f > 0.0:
        below_zero = np.where((bio_decay_factor <= 0.01) | (firmness <= senesc_f))[0]
    else:
        below_zero = np.where(bio_decay_factor <= 0.01)[0]
    if len(below_zero) > 0:
        point_b_day = float(t[below_zero[0]])
    else:
        point_b_day = float(t[-1])

    req_len = int(days / dt) + 1
    final_quality = float(quality[min(req_len - 1, len(quality) - 1)])
    point_a_day_int = int(round(point_a_day))
    point_b_day_int = int(round(point_b_day))
    commercial_remaining = max(0, point_a_day_int - int(days))
    biological_remaining = max(0, point_b_day_int - int(days))

    # Continuous arrays up to projection window for plotting
    plot_len = min(len(t), int((max(point_b_day, days) + 10) / dt))
    arrays_dict = {
        "t": t[:plot_len].tolist(),
        "quality": quality[:plot_len].tolist(),
        "firmness": firmness[:plot_len].tolist(),
        "brix": brix[:plot_len].tolist(),
        "mold": mold[:plot_len].tolist(),
        "cold_injury": cold_injury[:plot_len].tolist(),
        "ethylene": E_total[:plot_len].tolist(),
        "temperature": T_c_ext[:plot_len].tolist(),
        "humidity": RH_pct_ext[:plot_len].tolist(),
        "point_a_day": float(point_a_day),
        "point_b_day": float(point_b_day),
        "point_a_day_int": point_a_day_int,
        "point_b_day_int": point_b_day_int,
        "min_quality_threshold": min_quality,
        "mold_limit_threshold": mold_limit
    }

    return final_quality, commercial_remaining, biological_remaining, float(firmness[req_len - 1]), float(brix[req_len - 1]), float(point_a_day), float(point_b_day), arrays_dict

# =============================================================================
# 4. MODEL 2: WITHOUT ETHYLENE (Sofia Machado - 4D Stakeholder Quality)
# =============================================================================

def run_simulation_sofia_machado(fruit_key: str, T_c: list[float], RH_pct: list[float], days: int,
                                 firmness_0_user: float, brix_0_user: float, acidity_0_user: float,
                                 packaging_methods: Optional[list[str]] = None,
                                 current_owner_type: Optional[str] = "Retailer (Grocery Store)",
                                 dt: float = 0.05, custom_preset: dict = None):
    R = 8.314

    def k_temp_scaling(Ea, T, Tref):
        return np.exp((-Ea / R) * (1/T - 1/Tref))

    def calc_vpd(T_cels, RH_p):
        es = 0.6108 * np.exp(17.27 * T_cels / (T_cels + 237.3))
        ea = es * (RH_p / 100.0)
        return es - ea

    p = custom_preset if custom_preset is not None else PRESETS[fruit_key]
    is_climacteric = bool(p.get("is_climacteric", True))
    CI_type = p.get("CI_type", "threshold")
    CI_zone_min = float(p.get("CI_zone_min", 2.0))
    CI_zone_max = float(p.get("CI_zone_max", 7.0))
    CI_zone_peak = float(p.get("CI_zone_peak", 5.0))
    CI_thr = float(p.get("CI_thr", 0.0))
    CI_rate = float(p.get("CI_rate", 0.0))
    CI_sl_stress = float(p.get("CI_sl_stress", 1.0))
    CI_max_penalty = float(p.get("CI_max_penalty", 0.0))

    st_cfg = get_stakeholder_config(fruit_key, current_owner_type)
    w_firm, w_ratio, w_brix, w_acid = st_cfg["w_4d"]
    min_quality = st_cfg["min_quality"]
    mold_limit = st_cfg["mold_limit"]

    if packaging_methods is None:
        packaging_methods = ['bulk'] * days
    else:
        packaging_methods = [pm if pm is not None else 'bulk' for pm in packaging_methods]

    # Projection Window (365 days max) repeating last known values
    projection_days = 365
    max_sim_days = days + projection_days
    t = np.arange(0, max_sim_days + dt * 0.5, dt)

    def stretch_with_last_value(arr):
        arr_rep = np.repeat(np.array(arr), int(1/dt))
        if len(arr_rep) < len(t):
            arr_rep = np.append(arr_rep, [arr_rep[-1]] * (len(t) - len(arr_rep)))
        return arr_rep[:len(t)]

    T_c_ext = stretch_with_last_value(T_c)
    RH_pct_ext = stretch_with_last_value(RH_pct)
    pkg_ext = stretch_with_last_value(packaging_methods)

    T_K = T_c_ext + 273.15
    Tref_K = p["Tref_C"] + 273.15

    # 1. Thermal Time & Maturation Factor
    TT = np.zeros_like(t)
    T_base = float(p.get("T_base", 0.0))
    for i in range(1, len(t)):
        TT[i] = TT[i-1] + max(0.0, T_c_ext[i-1] - T_base) * dt

    TT_ref = float(p.get("SL_ref", 30)) * max(1.0, float(p.get("Tref_C", 1.0)) - T_base)

    # 2. VPD
    VPD = calc_vpd(T_c_ext, RH_pct_ext)
    VPD_ref = calc_vpd(p["Tref_C"], p["RH_ref"])

    # 3. Firmness, Brix, Acidity ODEs
    kT_firm = p["k_firm_ref"] * k_temp_scaling(p["Ea_J"], T_K, Tref_K)
    firmness = np.zeros_like(t)
    firmness_min = float(p["firmness_min"])
    firmness[0] = max(firmness_min + 1e-6, float(firmness_0_user))

    brix = np.zeros_like(t)
    brix_min = float(p["brix_min"])
    brix_max = float(p["brix_max"])
    brix[0] = float(brix_0_user)
    r0 = float(p.get("brix_g", 0.1))
    rT = k_temp_scaling(52000, T_K, Tref_K)

    acidity = np.zeros_like(t)
    acidity_min = min(float(p.get("acidity_min", 0.2)), float(acidity_0_user) * 0.5)
    acidity[0] = max(acidity_min + 1e-6, float(acidity_0_user))
    kT_acidity = float(p.get("k_acidity_ref", 0.02)) * k_temp_scaling(float(p.get("Ea_acidity_J", 55000)), T_K, Tref_K)

    SL_ref = float(p.get("SL_ref", 30))
    consumed_SL = np.zeros_like(t)
    cold_injury = np.zeros_like(t)

    for i in range(1, len(t)):
        # Acumulação de Dano por Frio (Cold Injury)
        if CI_type == "zone":
            curr_T = T_c_ext[i-1]
            if CI_zone_min <= curr_T <= CI_zone_max:
                if curr_T <= CI_zone_peak:
                    ratio = (curr_T - CI_zone_min) / max(1e-3, (CI_zone_peak - CI_zone_min))
                else:
                    ratio = (CI_zone_max - curr_T) / max(1e-3, (CI_zone_max - CI_zone_peak))
                delta_T_cold = max(0.0, ratio) * (CI_zone_max - CI_zone_min) * 0.5
            else:
                delta_T_cold = 0.0
        else:
            delta_T_cold = max(0.0, CI_thr - T_c_ext[i-1])
        dCI = CI_rate * delta_T_cold * dt
        cold_injury[i] = min(1.0, cold_injury[i-1] + dCI)

        VPD_excess = max(0, VPD[i-1] - VPD_ref)
        pkg_method = pkg_ext[i-1]
        fator_embalagem = PACKAGING_FACTORS.get(pkg_method, 1.0)
        VPD_efetivo = VPD_excess * fator_embalagem
        resp_factor = 0.5 + 0.5 * fator_embalagem
        mat_factor = (1.0 + 0.2 * min(2.0, TT[i-1] / max(1e-6, TT_ref))) if is_climacteric else 1.0

        # Firmness
        k_VPD_firm = 1.0 + p.get("beta_RH", 1.0) * VPD_efetivo
        dD = (-kT_firm[i-1] * resp_factor * k_VPD_firm * (firmness[i-1] - firmness_min)) * dt
        firmness[i] = max(firmness_min, firmness[i-1] + dD)

        # Brix
        if is_climacteric:
            r_VPD_brix = max(0, 1.0 - 0.2 * VPD_efetivo)
            r_brix_mod = r0 * rT[i-1] * r_VPD_brix * resp_factor * mat_factor
            x = max(0.01, brix[i-1] - brix_min)
            K = max(1e-6, (brix_max - brix_min))
            db = (r_brix_mod * x * (1.0 - x / K)) * dt
            brix[i] = min(brix_max, max(brix_min, brix[i-1] + db))
        else:
            r_resp = 0.005 * rT[i-1] * resp_factor
            db = -r_resp * dt
            brix[i] = max(brix_min, brix[i-1] + db)

        # Acidity
        dA = (-kT_acidity[i-1] * resp_factor * mat_factor * (acidity[i-1] - acidity_min)) * dt
        acidity[i] = max(acidity_min, acidity[i-1] + dA)

        # Physical Shelf Life Cap (with Cold Stress Factor)
        r_T_SL = k_temp_scaling(float(p.get("Ea_SL_J", 45000)), T_K[i-1], Tref_K) * resp_factor
        r_VPD_SL = 1.0 + 0.5 * VPD_efetivo
        ci_stress_factor = 1.0 + (CI_sl_stress - 1.0) * cold_injury[i-1]
        consumed_SL[i] = consumed_SL[i-1] + (r_T_SL * r_VPD_SL * ci_stress_factor) * dt

    remaining_SL_fisica_cap = np.maximum(0, SL_ref - consumed_SL)

    # 4. Mold Development (via VPD Deficit)
    RH_mold_thr = float(p["RH_mold_thr"])
    mold_rate_ref = float(p["mold_rate_ref"])
    mold_sens_RH = float(p["mold_sens_RH"])
    mold_max_penalty = float(p["mold_max_penalty"])
    Ea_mold_J = float(p["Ea_mold_J"])
    mold_T = k_temp_scaling(Ea_mold_J, T_K, Tref_K)

    mold = np.zeros_like(t)
    for i in range(1, len(t)):
        VPD_thr = calc_vpd(T_c_ext[i-1], RH_mold_thr)
        VPD_deficit = max(VPD_thr - VPD[i-1], 0)
        VPD_factor = 1.0 - np.exp(-mold_sens_RH * VPD_deficit * 5.0)
        cold_mold_inhibition = np.clip((T_c_ext[i-1] - (-1.0)) / 10.0, 0.05, 1.0)
        rate = mold_rate_ref * mold_T[i-1] * VPD_factor * cold_mold_inhibition
        dm = (rate * (1.0 - mold[i-1])) * dt
        mold[i] = min(1.0, max(0.0, mold[i-1] + dm))

    mold_penalty = mold_max_penalty * mold

    # 5. Continuous Smooth Quality (0-100) & Biological Sanity Synchronization
    firm_score = 1.0 / (1.0 + np.exp(-0.35 * (firmness - float(p["qual_firmness_threshold"]))))
    target_b = float(p["qual_brix_target"])
    brix_score = np.where(brix >= target_b, 1.0, np.exp(-((brix - target_b)**2) / 8.0))
    target_a = float(p.get("qual_acidity_target", 1.0))
    acidity_score = np.exp(-((acidity - target_a)**2) / 0.5)
    maturation_index = brix / np.maximum(1e-4, acidity)
    target_ratio = target_b / target_a
    ratio_score = np.where(maturation_index >= target_ratio, 1.0, np.exp(-((maturation_index - target_ratio)**2) / 50.0))

    group = get_fruit_group(fruit_key)
    if group in ["crunchy_texture", "climacteric_tactile", "juicy_flavor", "citrus"]:
        texture_factor = np.minimum(1.0, 2.0 * firm_score)
    else:
        texture_factor = 1.0
    quality_raw = 100.0 * (w_firm * firm_score + w_ratio * ratio_score + w_brix * brix_score + w_acid * acidity_score) * texture_factor
    
    # Fator de Sanidade Biológica: atinge 0.0 quando o bolor atinge o limiar crítico de 50%
    sanity_factor = np.maximum(0.0, 1.0 - (mold / 0.50))
    
    # Fator de Senescência Fisiológica: atinge 0.0 quando a shelf-life física de referência é esgotada
    cap_decay = np.clip(remaining_SL_fisica_cap / max(1.0, SL_ref * 0.10), 0.0, 1.0)

    # Fator de Penalização de Frio (Cold Injury)
    ci_factor = np.maximum(0.0, 1.0 - (CI_max_penalty * cold_injury))
    
    # Qualidade Global Integrada
    quality = np.maximum(0.0, np.minimum(100.0, quality_raw * sanity_factor * cap_decay * ci_factor))

    # 6. Milestone Points Calculation
    # Point A: Commercial Life Limit (The last day the fruit meets commercial standards before permanent expiration)
    marketable = (quality >= min_quality) & (mold <= mold_limit) & (remaining_SL_fisica_cap > 0)
    ok_indices = np.where(marketable)[0]
    if len(ok_indices) > 0:
        last_ok_idx = ok_indices[-1]
        point_a_day = float(t[last_ok_idx])
    else:
        point_a_day = 0.0

    # Point B: Biological Decay / Rotting (Senescência celular, bolor crítico ou colapso interno)
    bio_decay_factor = sanity_factor * cap_decay * ci_factor
    senesc_f = float(p.get("senescent_firmness_collapse", 8.5 if fruit_key == "pear" else 0.0))
    if senesc_f > 0.0:
        below_zero = np.where((bio_decay_factor <= 0.01) | (firmness <= senesc_f))[0]
    else:
        below_zero = np.where(bio_decay_factor <= 0.01)[0]
    if len(below_zero) > 0:
        point_b_day = float(t[below_zero[0]])
    else:
        point_b_day = float(t[-1])

    req_len = int(days / dt) + 1
    final_quality = float(quality[min(req_len - 1, len(quality) - 1)])
    point_a_day_int = int(round(point_a_day))
    point_b_day_int = int(round(point_b_day))
    commercial_remaining = max(0, point_a_day_int - int(days))
    biological_remaining = max(0, point_b_day_int - int(days))

    # Continuous arrays up to projection window for plotting
    plot_len = min(len(t), int((max(point_b_day, days) + 10) / dt))
    arrays_dict = {
        "t": t[:plot_len].tolist(),
        "quality": quality[:plot_len].tolist(),
        "firmness": firmness[:plot_len].tolist(),
        "brix": brix[:plot_len].tolist(),
        "acidity": acidity[:plot_len].tolist(),
        "ratio": maturation_index[:plot_len].tolist(),
        "mold": mold[:plot_len].tolist(),
        "cold_injury": cold_injury[:plot_len].tolist(),
        "temperature": T_c_ext[:plot_len].tolist(),
        "humidity": RH_pct_ext[:plot_len].tolist(),
        "point_a_day": float(point_a_day),
        "point_b_day": float(point_b_day),
        "point_a_day_int": point_a_day_int,
        "point_b_day_int": point_b_day_int,
        "min_quality_threshold": min_quality,
        "mold_limit_threshold": mold_limit
    }

    return final_quality, commercial_remaining, biological_remaining, float(firmness[req_len - 1]), float(brix[req_len - 1]), float(point_a_day), float(point_b_day), arrays_dict

# =============================================================================
# 5. FASTAPI SCHEMAS & ENDPOINTS
# =============================================================================

app = FastAPI()

_inference_lock = threading.Lock()
_status_lock = threading.Lock()
_current_operation = {
    "operation": None,
    "client_id": None,
    "store_id": None,
    "product_id": None,
    "algorithm": None,
}

class InitialMetrics(BaseModel):
    soluble_solids_brix: Optional[float] = None
    caliber_mm: Optional[float] = None
    quality_score: Optional[int] = None
    waste_kg: Optional[float] = None
    expiration_date: Optional[str] = None
    firmness: Optional[float] = None
    acidity: Optional[float] = None

class LotIdentification(BaseModel):
    lot_id: int
    batch_id: str
    culture_name: str
    fruit_type: str
    producer: str
    current_owner: str
    current_owner_type: Optional[str] = "Retailer (Grocery Store)"
    harvest_date: str
    initial_quantity_kg: float
    current_stock_kg: float
    delivered_quantity_kg: float
    initial_metrics: InitialMetrics

class DailyReading(BaseModel):
    date: str
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[float] = None
    ethylene_ppm: Optional[float] = None
    source: str = "SENSOR"

class WarehouseHistory(BaseModel):
    warehouse_id: int
    warehouse_location: str
    region: str
    meteo_source: str
    total_days_recorded: int
    starting_date: Optional[str] = None
    packaging_method: Optional[str] = "bulk"
    daily_readings: List[DailyReading] = []

class LifecycleDataRequest(BaseModel):
    version: str = "1.0"
    export_metadata: Dict[str, Any] = {}
    lot_identification: LotIdentification
    plantation_origin: Dict[str, Any] = {}
    plantation_agricultural_events: List[Any] = []
    current_warehouse: Dict[str, Any] = {}
    meteorology_and_imputation_strategy: Dict[str, Any] = {}
    blockchain_ledger: Dict[str, Any] = {}
    transport_and_logistics: List[Any] = []
    sensor_history_by_warehouse: List[WarehouseHistory]
    plot_info: bool = True

class ForecastResponse(BaseModel):
    service: Literal["inference"] = "inference"
    client_id: str
    store_id: str
    algorithm: Literal["ode_academic", "ode_new"]
    product_id: str
    quality_index: float
    commercial_lifetime_remaining: float
    biological_lifetime_remaining: float
    point_a_day: float
    point_b_day: float
    firmness: float
    brix: float
    continuous_data: Optional[Dict[str, Any]] = None

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: LifecycleDataRequest, client_id: str = "streamlit-ui", store_id: str = "store-1"):
    fruit_type = request.lot_identification.fruit_type.lower()
    culture_name = request.lot_identification.culture_name.lower()
    raw_key = f"{fruit_type}_{culture_name}"
    
    if raw_key in PRESETS:
        fruit_key = raw_key
    elif fruit_type in PRESETS:
        fruit_key = fruit_type
    elif culture_name in PRESETS:
        fruit_key = culture_name
    else:
        # Fallback to closest match or kiwi_hayward
        fruit_key = "kiwi_hayward"
        for k in PRESETS:
            if k in fruit_type or k in culture_name:
                fruit_key = k
                break

    preset = PRESETS[fruit_key]
    owner_type = request.lot_identification.current_owner_type or "Retailer (Grocery Store)"

    acquired = _inference_lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse(status_code=429, content={"detail": "Inference busy"})

    try:
        T_c = []
        RH_pct = []
        E_ppm = []
        region_codes = []
        packaging_methods = []
        start_date_str = None

        for wh in request.sensor_history_by_warehouse:
            if not wh.daily_readings and wh.starting_date and wh.total_days_recorded > 0:
                s_dt = datetime.strptime(wh.starting_date, "%Y-%m-%d")
                for j in range(wh.total_days_recorded):
                    c_dt = s_dt + timedelta(days=j)
                    wh.daily_readings.append(DailyReading(date=c_dt.strftime("%Y-%m-%d"), source="GENERATED"))

            for reading in wh.daily_readings:
                if start_date_str is None:
                    start_date_str = reading.date
                T_c.append(reading.temperature_celsius)
                RH_pct.append(reading.humidity_percent)
                E_ppm.append(reading.ethylene_ppm)
                region_codes.append(wh.region or "PT-LVT")
                packaging_methods.append(wh.packaging_method or "bulk")

        if not T_c:
            return JSONResponse(status_code=400, content={"detail": "No sensor readings provided."})

        days = len(T_c)
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else datetime.today()

        # Imputation Strategy
        fallback_mode = request.meteorology_and_imputation_strategy.get("json_fallback_mode", "IPMA")
        if fallback_mode == "FIXED":
            fixed_t = float(request.meteorology_and_imputation_strategy.get("fixed_temperature_celsius", 4.0))
            fixed_rh = float(request.meteorology_and_imputation_strategy.get("fixed_humidity_percent", 90.0))
            fixed_eth = float(request.meteorology_and_imputation_strategy.get("fixed_ethylene_ppm", 0.05))
            T_c = [fixed_t if t is None else t for t in T_c]
            RH_pct = [fixed_rh if rh is None else rh for rh in RH_pct]
            E_ppm_filled = [fixed_eth if e is None else e for e in E_ppm]
        else:
            T_c, RH_pct = fill_nulls_with_warehouse_sim(T_c, RH_pct, region_codes, alphas=[0.9]*days, start_date=start_date)
            E_ppm_filled = [0.0 if e is None else e for e in E_ppm]

        # Model Decision Trigger: If at least 1 reading of ethylene exists (> 0), use Prof. Luís Paulo
        has_any_ethylene = any(e is not None and e > 0 for e in E_ppm)

        b0 = request.lot_identification.initial_metrics.soluble_solids_brix or preset["brix_0_default"]
        f0 = request.lot_identification.initial_metrics.firmness or preset["firmness_0_default"]
        a0 = request.lot_identification.initial_metrics.acidity or preset.get("acidity_0_default", 0.5)

        if has_any_ethylene:
            algo = "ode_academic"
            q, comm_life, bio_life, final_f, final_b, pt_a, pt_b, arrays_dict = run_simulation_prof_luis_paulo(
                fruit_key=fruit_key,
                T_c=T_c,
                E_ext_ppm=E_ppm_filled,
                RH_pct=RH_pct,
                days=days,
                firmness_0_user=f0,
                brix_0_user=b0,
                packaging_methods=packaging_methods,
                current_owner_type=owner_type
            )
        else:
            algo = "ode_new"
            q, comm_life, bio_life, final_f, final_b, pt_a, pt_b, arrays_dict = run_simulation_sofia_machado(
                fruit_key=fruit_key,
                T_c=T_c,
                RH_pct=RH_pct,
                days=days,
                firmness_0_user=f0,
                brix_0_user=b0,
                acidity_0_user=a0,
                packaging_methods=packaging_methods,
                current_owner_type=owner_type
            )

        return ForecastResponse(
            service="inference",
            client_id=client_id,
            store_id=store_id,
            algorithm=algo,
            product_id=fruit_key,
            quality_index=q,
            commercial_lifetime_remaining=comm_life,
            biological_lifetime_remaining=bio_life,
            point_a_day=pt_a,
            point_b_day=pt_b,
            firmness=final_f,
            brix=final_b,
            continuous_data=arrays_dict if request.plot_info else None
        )
    finally:
        _inference_lock.release()

@app.get("/presets")
def get_presets():
    return {"fruits": list(PRESETS.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8181)
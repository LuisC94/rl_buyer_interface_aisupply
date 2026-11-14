"""
Motor Lifecycle (LC v0.2.0) embutido no Buyer Agent, em formato de passo diário compilado (Numba JIT).

As equações são as de `engine/lifecycle/lc_v0_2_0.py` (cópia exata do Lifecycle Web Service):
  - Modelo 1 (com etileno)  -> run_simulation_prof_luis_paulo
  - Modelo 2 (sem etileno)  -> run_simulation_sofia_machado

O LC original integra a série completa de uma vez (Euler, dt=0.05). Aqui o mesmo esquema de Euler
é decomposto em passos de 1 dia (20 sub-passos), com o clima constante dentro do dia, que é
exatamente o que o LC faz ao "esticar" cada leitura diária por 1/dt sub-passos. Isto permite que o
ambiente de treino avance cada lote um dia de cada vez sem chamar o Web Service.

Para atualizar o modelo: copiar o novo lc_v0_2_0.py do Lifecycle Web Service para engine/lifecycle/
(os presets e configurações por stakeholder são lidos diretamente desse ficheiro).
"""
import math
import os
import sys

import numba
import numpy as np

_LC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifecycle")
if _LC_DIR not in sys.path:
    sys.path.insert(0, _LC_DIR)

from lc_v0_2_0 import (  # noqa: E402
    PRESETS as LC_PRESETS, PACKAGING_FACTORS as PACKAGING_FACTORS_LC, get_stakeholder_config, get_fruit_group,
)

DT = 0.05
SUBSTEPS = 20  # int(1/DT)

# Índices do vetor de estado de um lote
S_FIRM, S_BRIX, S_ACID, S_MOLD, S_EINT, S_AGE, S_CI, S_SL, S_TT = range(9)
STATE_SIZE = 9

TEXTURE_GROUPS = ("crunchy_texture", "climacteric_tactile", "juicy_flavor", "citrus")

# Aliases usados pelo ambiente DSAC original (FruitModel2) -> chaves LC v0.2.0
ENV_ALIASES = {
    "maca_gala": "apple_gala",
    "maca_fuji": "apple_fuji",
    "maca_golden": "apple_golden",
    "maca_reineta": "apple_reineta",
}


def resolve_lc_fruit_key(fruit: str) -> str:
    """Mesma lógica de resolve_fruit_key() do Lifecycle Web Service (main.py), mais os aliases do ambiente DSAC."""
    s = str(fruit or "").lower().strip()
    if s in LC_PRESETS:
        return s
    if s in ENV_ALIASES:
        return ENV_ALIASES[s]
    rules = [
        (("hayward",), "kiwi_hayward"), (("baby",), "kiwi_baby"), (("kiwi", "gold"), "kiwi_gold"),
        (("kiwi",), "kiwi_hayward"), (("gala",), "apple_gala"), (("fuji",), "apple_fuji"),
        (("golden",), "apple_golden"), (("reineta",), "apple_reineta"), (("granny",), "apple_golden"),
        (("smith",), "apple_golden"), (("red delicious",), "apple_red_delicious"),
        (("esmolfe",), "apple_reineta"), (("bravo",), "apple_reineta"), (("pink lady",), "apple_fuji"),
        (("alcobaca",), "apple_gala"), (("alcobaça",), "apple_gala"), (("jonagold",), "apple_golden"),
        (("royal",), "apple_gala"), (("maca",), "apple_gala"), (("maçã",), "apple_gala"),
        (("apple",), "apple_gala"), (("pera",), "pear"), (("pêra",), "pear"), (("pear",), "pear"),
        (("rocha",), "pear"), (("morango",), "strawberry"), (("strawberry",), "strawberry"),
        (("framboesa",), "raspberry"), (("raspberry",), "raspberry"), (("figo",), "fig"), (("fig",), "fig"),
        (("mirtilo",), "blueberry"), (("blueberry",), "blueberry"), (("cereja",), "cherry"),
        (("cherry",), "cherry"), (("banana",), "banana"), (("pessego",), "peach"), (("pêssego",), "peach"),
        (("peach",), "peach"), (("ameixa",), "plum"), (("plum",), "plum"), (("melao",), "melon"),
        (("melão",), "melon"), (("melon",), "melon"), (("uva",), "grape"), (("grape",), "grape"),
        (("laranja",), "orange"), (("orange",), "orange"),
    ]
    for words, key in rules:
        if all(w in s for w in words):
            return key
    for k in LC_PRESETS:
        if k in s:
            return k
    return "apple_gala"


def build_param_array(fruit_key: str, owner_type: str = "Retailer") -> np.ndarray:
    """Converte o preset LC + configuração do stakeholder num vetor float64 para o kernel JIT."""
    p = LC_PRESETS[fruit_key]
    st = get_stakeholder_config(fruit_key, owner_type)
    w_firm2, w_brix2 = st["w_2d"]
    w4 = st["w_4d"]
    acidity_0 = float(p.get("acidity_0_default", 1.0))
    acidity_min = min(float(p.get("acidity_min", 0.2)), acidity_0 * 0.5)
    T_base = float(p.get("T_base", 0.0))
    SL_ref_m2 = float(p.get("SL_ref", 30))
    TT_ref = SL_ref_m2 * max(1.0, float(p.get("Tref_C", 1.0)) - T_base)
    return np.array([
        float(p["Tref_C"]),                          # 0
        float(p["Ea_J"]),                            # 1
        float(p["k_firm_ref"]),                      # 2
        float(p.get("beta_RH", 1.0)),                # 3
        float(p["RH_ref"]),                          # 4
        float(p["brix_min"]),                        # 5
        float(p["brix_max"]),                        # 6
        float(p.get("brix_g", 0.1)),                 # 7
        float(p["qual_brix_target"]),                # 8
        float(p["firmness_min"]),                    # 9
        float(p["qual_firmness_threshold"]),         # 10
        acidity_min,                                 # 11
        float(p.get("k_acidity_ref", 0.02)),         # 12
        float(p.get("Ea_acidity_J", 55000)),         # 13
        float(p.get("qual_acidity_target", 1.0)),    # 14
        float(p.get("SL_ref", 60.0)),                # 15 (modelo 1)
        float(p.get("Ea_SL_J", 45000)),              # 16
        float(p["E0_int"]) if "E0_int" in p else 0.01,  # 17
        float(p.get("Eref_prod", 0.12)),             # 18
        float(p.get("E_t0", 10.0)),                  # 19
        float(p.get("E_g", 0.9)),                    # 20
        float(p.get("E_auto", 0.35)),                # 21
        float(p.get("E_decay", 0.7)),                # 22
        float(p.get("Ea_E_J", 52000)),               # 23
        float(p.get("E_ext_shift", 2.0)),            # 24
        float(p.get("alpha_E", 0.1)),                # 25
        float(p["RH_mold_thr"]),                     # 26
        float(p["mold_rate_ref"]),                   # 27
        float(p["mold_sens_RH"]),                    # 28
        float(p["Ea_mold_J"]),                       # 29
        1.0 if bool(p.get("is_climacteric", True)) else 0.0,  # 30
        1.0 if p.get("CI_type", "threshold") == "zone" else 0.0,  # 31
        float(p.get("CI_zone_min", 2.0)),            # 32
        float(p.get("CI_zone_max", 7.0)),            # 33
        float(p.get("CI_zone_peak", 5.0)),           # 34
        float(p.get("CI_thr", 0.0)),                 # 35
        float(p.get("CI_rate", 0.0)),                # 36
        float(p.get("CI_sl_stress", 1.0)),           # 37
        float(p.get("CI_max_penalty", 0.0)),         # 38
        1.0 if get_fruit_group(fruit_key) in TEXTURE_GROUPS else 0.0,  # 39
        float(w_firm2), float(w_brix2),              # 40, 41
        float(w4[0]), float(w4[1]), float(w4[2]), float(w4[3]),  # 42..45 (firm, ratio, brix, acid)
        T_base,                                      # 46
        SL_ref_m2,                                   # 47 (modelo 2)
        TT_ref,                                      # 48
        float(st["min_quality"]),                    # 49
        float(st["mold_limit"]),                     # 50
    ], dtype=np.float64)


def initial_state(fruit_key: str, firmness_0=None, brix_0=None, acidity_0=None) -> np.ndarray:
    """Estado inicial de um lote acabado de chegar (t = 0 no LC)."""
    p = LC_PRESETS[fruit_key]
    f0 = float(firmness_0 if firmness_0 is not None else p["firmness_0_default"])
    b0 = float(brix_0 if brix_0 is not None else p["brix_0_default"])
    a0 = float(acidity_0 if acidity_0 is not None else p.get("acidity_0_default", 1.0))
    acidity_min = min(float(p.get("acidity_min", 0.2)), a0 * 0.5)
    s = np.zeros(STATE_SIZE, dtype=np.float64)
    s[S_FIRM] = max(float(p["firmness_min"]) + 1e-6, f0)
    s[S_BRIX] = b0
    s[S_ACID] = max(acidity_min + 1e-6, a0)
    s[S_EINT] = float(p["E0_int"]) if "E0_int" in p else 0.01
    return s


# ---------------------------------------------------------------------------------------------
# Kernels JIT
# ---------------------------------------------------------------------------------------------

@numba.njit(cache=True, fastmath=False)
def _k_temp(Ea, T_K, Tref_K):
    return math.exp((-Ea / 8.314) * (1.0 / T_K - 1.0 / Tref_K))


@numba.njit(cache=True, fastmath=False)
def _vpd(T_c, RH):
    es = 0.6108 * math.exp(17.27 * T_c / (T_c + 237.3))
    return es - es * (RH / 100.0)


@numba.njit(cache=True, fastmath=False)
def _cold_injury_delta(T_c, P):
    if P[31] > 0.5:  # zona (fruta de caroço)
        zmin, zmax, zpeak = P[32], P[33], P[34]
        if zmin <= T_c <= zmax:
            if T_c <= zpeak:
                ratio = (T_c - zmin) / max(1e-3, (zpeak - zmin))
            else:
                ratio = (zmax - T_c) / max(1e-3, (zmax - zpeak))
            return max(0.0, ratio) * (zmax - zmin) * 0.5
        return 0.0
    return max(0.0, P[35] - T_c)


@numba.njit(cache=True, fastmath=False)
def lc_quality(s, model, P):
    """Qualidade integrada (0-100) do LC v0.2.0 para o estado s."""
    firm_score = 1.0 / (1.0 + math.exp(-0.35 * (s[S_FIRM] - P[10])))
    tb = P[8]
    brix_score = 1.0 if s[S_BRIX] >= tb else math.exp(-((s[S_BRIX] - tb) ** 2) / 8.0)
    texture = min(1.0, 2.0 * firm_score) if P[39] > 0.5 else 1.0
    if model == 1:
        SL_ref = P[15]
        quality_raw = 100.0 * (P[40] * firm_score + P[41] * brix_score) * texture
    else:
        SL_ref = P[47]
        ta = P[14]
        acidity_score = math.exp(-((s[S_ACID] - ta) ** 2) / 0.5)
        mi = s[S_BRIX] / max(1e-4, s[S_ACID])
        tr = tb / ta
        ratio_score = 1.0 if mi >= tr else math.exp(-((mi - tr) ** 2) / 50.0)
        quality_raw = 100.0 * (P[42] * firm_score + P[43] * ratio_score + P[44] * brix_score + P[45] * acidity_score) * texture
    remaining_sl = max(0.0, SL_ref - s[S_SL])
    sanity = max(0.0, 1.0 - (s[S_MOLD] / 0.50))
    cap_decay = min(1.0, max(0.0, remaining_sl / max(1.0, SL_ref * 0.10)))
    ci_factor = max(0.0, 1.0 - (P[38] * s[S_CI]))
    return max(0.0, min(100.0, quality_raw * sanity * cap_decay * ci_factor))


@numba.njit(cache=True, fastmath=False)
def lc_advance_day(s, T_c, RH, E_ext, pkg_factor, model, P):
    """Avança o estado de um lote 1 dia (20 sub-passos Euler, dt=0.05). Devolve (novo_estado, qualidade)."""
    out = s.copy()
    T_K = T_c + 273.15
    Tref_K = P[0] + 273.15
    resp = 0.5 + 0.5 * pkg_factor
    climacteric = P[30] > 0.5
    RH_deficit = max(0.0, (P[4] - RH) / 100.0)
    kT_firm = P[2] * _k_temp(P[1], T_K, Tref_K)
    mold_T = _k_temp(P[29], T_K, Tref_K)
    r_T_SL = _k_temp(P[16], T_K, Tref_K) * resp
    cold_inhib = min(1.0, max(0.05, (T_c - (-1.0)) / 10.0))
    dCI = P[36] * _cold_injury_delta(T_c, P) * DT

    if model == 1:
        prod_T = _k_temp(P[23], T_K, Tref_K)
        t0_eff = P[19] - P[24] * math.log1p(max(0.0, E_ext))
        RH_excess = max(0.0, (RH - P[26]) / 100.0)
        RH_factor = 1.0 - math.exp(-P[28] * RH_excess)
        mold_rate = P[27] * mold_T * RH_factor * cold_inhib
        r_RH_SL = 1.0 + 0.5 * RH_deficit * pkg_factor
        kRH = 1.0 + P[3] * RH_deficit * pkg_factor
        bRH = RH_deficit * pkg_factor
        rRH = 1.0 - 0.6 * bRH
        for _ in range(SUBSTEPS):
            firm, brix, mold, E_int, age, ci, sl = out[S_FIRM], out[S_BRIX], out[S_MOLD], out[S_EINT], out[S_AGE], out[S_CI], out[S_SL]
            # 1. Etileno endógeno
            ramp = 1.0 / (1.0 + math.exp(-(P[20] * (age - t0_eff))))
            prod = P[18] * prod_T * ramp
            dE = (prod * (1.0 + P[21] * E_int) - P[22] * E_int) * DT
            E_total = E_ext + E_int
            # 2. Firmeza
            k = kT_firm * resp * kRH * (1.0 + P[25] * E_total)
            new_firm = max(P[9], firm + (-k * (firm - P[9])) * DT)
            # 3. Brix
            if climacteric:
                r = P[7] * prod_T * rRH * resp * (1.0 + 0.25 * E_total)
                x = max(0.01, brix - P[5])
                K = max(1e-6, (P[6] - P[5]))
                new_brix = min(P[6], max(P[5], brix + (r * x * (1.0 - x / K)) * DT))
            else:
                new_brix = max(P[5], brix + (-0.005 * prod_T * resp) * DT)
            # 4. Dano por frio, bolor e shelf-life física
            out[S_CI] = min(1.0, ci + dCI)
            out[S_MOLD] = min(1.0, max(0.0, mold + (mold_rate * (1.0 - mold)) * DT))
            ci_stress = 1.0 + (P[37] - 1.0) * ci
            out[S_SL] = sl + (r_T_SL * r_RH_SL * ci_stress) * DT
            out[S_EINT] = max(0.0, E_int + dE)
            out[S_FIRM] = new_firm
            out[S_BRIX] = new_brix
            out[S_AGE] = age + DT
    else:
        rT = _k_temp(52000.0, T_K, Tref_K)
        kT_acid = P[12] * _k_temp(P[13], T_K, Tref_K)
        VPD = _vpd(T_c, RH)
        VPD_ref = _vpd(P[0], P[4])
        VPD_eff = max(0.0, VPD - VPD_ref) * pkg_factor
        k_VPD_firm = 1.0 + P[3] * VPD_eff
        r_VPD_brix = max(0.0, 1.0 - 0.2 * VPD_eff)
        r_VPD_SL = 1.0 + 0.5 * VPD_eff
        VPD_thr = _vpd(T_c, P[26])
        VPD_def = max(VPD_thr - VPD, 0.0)
        VPD_factor = 1.0 - math.exp(-P[28] * VPD_def * 5.0)
        mold_rate = P[27] * mold_T * VPD_factor * cold_inhib
        dTT = max(0.0, T_c - P[46]) * DT
        for _ in range(SUBSTEPS):
            firm, brix, acid, mold, age, ci, sl, TT = out[S_FIRM], out[S_BRIX], out[S_ACID], out[S_MOLD], out[S_AGE], out[S_CI], out[S_SL], out[S_TT]
            mat = (1.0 + 0.2 * min(2.0, TT / max(1e-6, P[48]))) if climacteric else 1.0
            new_firm = max(P[9], firm + (-kT_firm * resp * k_VPD_firm * (firm - P[9])) * DT)
            if climacteric:
                r = P[7] * rT * r_VPD_brix * resp * mat
                x = max(0.01, brix - P[5])
                K = max(1e-6, (P[6] - P[5]))
                new_brix = min(P[6], max(P[5], brix + (r * x * (1.0 - x / K)) * DT))
            else:
                new_brix = max(P[5], brix + (-0.005 * rT * resp) * DT)
            new_acid = max(P[11], acid + (-kT_acid * resp * mat * (acid - P[11])) * DT)
            ci_stress = 1.0 + (P[37] - 1.0) * ci
            out[S_CI] = min(1.0, ci + dCI)
            out[S_SL] = sl + (r_T_SL * r_VPD_SL * ci_stress) * DT
            out[S_MOLD] = min(1.0, max(0.0, mold + (mold_rate * (1.0 - mold)) * DT))
            out[S_TT] = TT + dTT
            out[S_FIRM] = new_firm
            out[S_BRIX] = new_brix
            out[S_ACID] = new_acid
            out[S_AGE] = age + DT
    return out, lc_quality(out, model, P)


@numba.njit(cache=True, fastmath=False)
def lc_project_rsl(s, quality, T_c, RH, E_ext, pkg_factor, model, P, quality_floor, max_days):
    """Dias até a qualidade cair abaixo de quality_floor mantendo o clima atual (mesma regra do ambiente DSAC)."""
    days = 0
    cur = s.copy()
    q = quality
    while days < max_days:
        if q < quality_floor:
            break
        cur, q = lc_advance_day(cur, T_c, RH, E_ext, pkg_factor, model, P)
        days += 1
    return days

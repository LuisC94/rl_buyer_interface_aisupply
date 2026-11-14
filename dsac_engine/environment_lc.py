"""
Ambiente DSAC original (engine/reference/environment_constrained.py, cópia exata) com o motor
biológico FruitModel2 substituído pelo Lifecycle LC v0.2.0 (engine/lc_kernel.py).

Só são redefinidos os pontos onde o ambiente original chama o motor biológico:
  - PRESETS (valores iniciais dos lotes: firmeza, brix, etileno)
  - advance_batch_one_day()  -> envelhece um lote 1 dia
  - project_batch_rsl()      -> dias até a qualidade cair abaixo de 30 (regra original do ambiente)
Custos, recompensa, FEFO, markdown e o vetor de estado (30 variáveis) ficam exatamente iguais.
"""
import importlib.util
import io
import os
import sys

import numpy as np
import pandas as pd

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

# Na app Streamlit existe outro environment_constrained.py na raiz (ambiente PPO antigo).
# Carrega-se o ficheiro original do DSAC pelo caminho, para nunca apanhar o módulo errado.
_spec = importlib.util.spec_from_file_location(
    "dsac_reference_environment_constrained", os.path.join(_REF_DIR, "environment_constrained.py"))
_ref_env_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ref_env_module
_spec.loader.exec_module(_ref_env_module)
StockEnvironment = _ref_env_module.StockEnvironment  # (ficheiro original)

from .lc_kernel import (  # noqa: E402
    LC_PRESETS, PACKAGING_FACTORS_LC, ENV_ALIASES, STATE_SIZE, S_FIRM, S_BRIX, S_ACID, S_MOLD, S_EINT, S_AGE,
    S_CI, S_SL, S_TT, build_param_array, initial_state, lc_advance_day, lc_quality, lc_project_rsl,
    resolve_lc_fruit_key,
)

QUALITY_FLOOR = 30.0      # mesma regra do ambiente original (lote estragado / RSL)
MAX_PROJECTION_DAYS = 60  # mesmo horizonte de projeção do ambiente original


def _env_presets():
    """Presets no formato que o ambiente original espera, gerados a partir dos presets LC v0.2.0."""
    out = {}
    for key, p in LC_PRESETS.items():
        out[key] = {
            "label": p.get("label", key),
            # Chaves usadas pelo ambiente original para criar lotes
            "dureza_0_default": float(p["firmness_0_default"]),
            "brix_0_default": float(p["brix_0_default"]),
            "E0_int": float(p.get("E0_int", 0.01)),
            # Chaves usadas apenas para construir o preset_array FruitModel2 (não usado pelo motor LC)
            "Tref_C": float(p["Tref_C"]), "Ea_J": float(p["Ea_J"]), "k_firm_ref": float(p["k_firm_ref"]),
            "alpha_E": float(p.get("alpha_E", 0.1)), "beta_RH": float(p.get("beta_RH", 1.0)),
            "RH_ref": float(p["RH_ref"]), "dureza_min": float(p["firmness_min"]),
            "brix_min": float(p["brix_min"]), "brix_max": float(p["brix_max"]), "brix_g": float(p.get("brix_g", 0.1)),
            "qual_firm_threshold": float(p["qual_firmness_threshold"]), "qual_brix_target": float(p["qual_brix_target"]),
            "Eref_prod": float(p.get("Eref_prod", 0.12)), "E_t0": float(p.get("E_t0", 10.0)),
            "E_g": float(p.get("E_g", 0.9)), "E_auto": float(p.get("E_auto", 0.35)),
            "E_decay": float(p.get("E_decay", 0.7)), "Ea_E_J": float(p.get("Ea_E_J", 52000)),
            "E_ext_shift": float(p.get("E_ext_shift", 2.0)), "RH_mold_thr": float(p["RH_mold_thr"]),
            "mold_rate_ref": float(p["mold_rate_ref"]), "mold_sens_RH": float(p["mold_sens_RH"]),
            "mold_max_penalty": float(p.get("mold_max_penalty", 0.65)), "Ea_mold_J": float(p["Ea_mold_J"]),
        }
    for alias, key in ENV_ALIASES.items():
        out[alias] = out[key]
    return out


def dataframe_to_source(df: pd.DataFrame):
    """O ambiente original lê com pd.read_excel(); passamos o DataFrame em memória como ficheiro Excel."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf


class StockEnvironmentLC(StockEnvironment):
    def __init__(self, data, *args, owner_type="Retailer", packaging="bulk", lc_model=None, **kwargs):
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            data = dataframe_to_source(frame)
        else:
            frame = pd.read_excel(data)
            data = dataframe_to_source(frame)

        # O ambiente original resolve a fruta pela coluna 'fruit_type' contra self.PRESETS
        self.lc_fruit_key = resolve_lc_fruit_key(frame["fruit_type"].iloc[0] if "fruit_type" in frame.columns else "")
        frame_fruit = self.lc_fruit_key
        if "fruit_type" in frame.columns:
            frame = frame.copy()
            frame["fruit_type"] = frame_fruit
            data = dataframe_to_source(frame)
        self.PRESETS = _env_presets()

        super().__init__(data, *args, **kwargs)

        self.owner_type = owner_type
        self.packaging = packaging
        self.pkg_factor = float(PACKAGING_FACTORS_LC.get(packaging, 1.0))
        # Mesmo critério do Lifecycle Web Service: com leituras de etileno > 0 usa o Modelo 1, senão o Modelo 2
        if lc_model is None:
            has_ethylene = "ethylene" in self.df.columns and bool((self.df["ethylene"].fillna(0) > 0).any())
            lc_model = 1 if has_ethylene else 2
        self.lc_model = int(lc_model)
        self.lc_params = build_param_array(self.lc_fruit_key, owner_type)

    # ------------------------------------------------------------------ estado LC de um lote
    def _lc_state(self, batch):
        s = batch.get("lc_state")
        if s is not None:
            return s
        # Lote criado pelo ambiente original (chegada / reset): estado inicial LC
        s = initial_state(self.lc_fruit_key, firmness_0=batch.get("dureza"), brix_0=batch.get("brix"))
        s[S_MOLD] = float(batch.get("mold", 0.0))
        s[S_EINT] = float(batch.get("E_int", s[S_EINT]))
        s[S_AGE] = float(batch.get("age", 0.0))
        return s

    def _today_climate(self):
        row = self.data.iloc[min(self.current_step, len(self.data) - 1)]
        T_c = float(row["temperature"] if "temperature" in row else 22.0)
        RH_pct = float(row["humidity"] if "humidity" in row else 70.0)
        E_ext = float(row["ethylene"] if "ethylene" in row else 0.8)
        return T_c, RH_pct, E_ext

    # ------------------------------------------------------------------ redefinições do motor biológico
    def advance_batch_one_day(self, batch, T_c, RH_pct, E_ext_ppm):
        s = self._lc_state(batch)
        s_next, quality = lc_advance_day(s, float(T_c), float(RH_pct), float(E_ext_ppm), self.pkg_factor,
                                         self.lc_model, self.lc_params)
        nb = dict(batch)
        nb.update({
            "lc_state": s_next,
            "dureza": float(s_next[S_FIRM]),
            "brix": float(s_next[S_BRIX]),
            "mold": float(s_next[S_MOLD]),
            "E_int": float(s_next[S_EINT]),
            "age": float(batch.get("age", 0.0)) + 1.0,
            "quality": float(quality),
        })
        nb.setdefault("lot_id", "LOT-1")
        nb.setdefault("initial_quantity", batch["quantity"])
        return nb

    def project_batch_rsl(self, batch):
        if not self.enable_lifecycle:
            return 60
        T_c, RH_pct, E_ext = self._today_climate()
        s = self._lc_state(batch)
        q = float(batch["quality"]) if "lc_state" in batch else float(lc_quality(s, self.lc_model, self.lc_params))
        return int(lc_project_rsl(s, q, T_c, RH_pct, E_ext, self.pkg_factor, self.lc_model, self.lc_params,
                                  QUALITY_FLOOR, MAX_PROJECTION_DAYS))

    # ------------------------------------------------------------------ utilitários para /decide
    def make_batch(self, quantity, age_days=0, climate_history=None):
        """Cria um lote com 'age_days' dias, envelhecido com o clima fornecido (lista de (T, RH, E))."""
        p = self.PRESETS[self.fruit_key]
        b = {
            "lot_id": f"LOT-{self._next_lot_id}", "harvest_date": self._get_current_date_str(),
            "initial_quantity": float(quantity), "quantity": float(quantity),
            "initial_dureza": p["dureza_0_default"], "dureza": p["dureza_0_default"],
            "initial_brix": p["brix_0_default"], "brix": p["brix_0_default"],
            "mold": 0.0, "E_int": p["E0_int"], "age": 0.0, "quality": 100.0, "daily_readings": [],
        }
        self._next_lot_id += 1
        clim = list(climate_history or [])
        for d in range(int(max(0, age_days))):
            T_c, RH_pct, E_ext = clim[d] if d < len(clim) else (clim[-1] if clim else self._today_climate())
            b = self.advance_batch_one_day(b, T_c, RH_pct, E_ext)
        if "lc_state" not in b:
            s = self._lc_state(b)
            b["lc_state"] = s
            b["quality"] = float(lc_quality(s, self.lc_model, self.lc_params))
        return b

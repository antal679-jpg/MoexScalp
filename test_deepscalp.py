"""
test_deepscalp.py  —  офлайн тест-харнесс для DeepScalp (Уровень 0).

Что проверяет, НЕ требуя TK_TOKEN, GPU и накопленных данных:
  A. Все сети из TkConfig.ini собираются конструктором TkModel.
  B. Оба автоэнкодера и прогнозная модель инстанцируются на CPU.
  C. Forward прогнозной модели проходит на входе шириной InputWidth (=20) и даёт код нужного размера.
  D. ЭМПИРИЧЕСКИ подтверждает баг train/serve: инференс собирает 19 признаков/шаг,
     а модель ждёт 20 -> вход 19xN отвергается. Это и есть рассинхрон из п.6.2 разбора.
  E. Численная корректность TkStatistics (объёмы, pivot) на ручных мок-данных.

ЗАПУСК из КОРНЯ репозитория DeepScalp (где лежит TkConfig.ini):
    python test_deepscalp.py

Зависимости: torch (хватит CPU-сборки), numpy, tinkoff-investments (как в самом проекте).
Идея CPU-патча: код проекта жёстко зовёт .cuda(); мы делаем .cuda() no-op, чтобы всё шло на CPU.
"""

import os
import sys
import json
import configparser
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Нейтрализуем CUDA: .cuda() на тензоре и модуле просто возвращает self (CPU)
# ---------------------------------------------------------------------------
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

sys.path.insert(0, os.getcwd())

CFG = configparser.ConfigParser()
if not CFG.read("TkConfig.ini"):
    print("ОШИБКА: запусти скрипт из корня репозитория DeepScalp (не найден TkConfig.ini).")
    sys.exit(2)

_results = []
def check(name, cond, detail=""):
    _results.append(bool(cond))
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))

# ---------------------------------------------------------------------------
# A. TkModel собирает все сети из конфига
# ---------------------------------------------------------------------------
try:
    from TkModules.TkModel import TkModel
    nets = {
        ("Autoencoders", "OrderbookEncoder"),
        ("Autoencoders", "OrderbookDecoder"),
        ("Autoencoders", "LastTradesEncoder"),
        ("Autoencoders", "LastTradesDecoder"),
        ("TimeSeries",   "MLP"),
        ("TimeSeries",   "AuxMLP"),
    }
    for section, key in nets:
        spec = json.loads(CFG[section][key])
        _ = TkModel(spec)
    check("A. TkModel строит все сети из конфига", True, f"{len(nets)} сетей")
except Exception as e:
    check("A. TkModel строит все сети из конфига", False, repr(e))

# ---------------------------------------------------------------------------
# B. Автоэнкодеры инстанцируются
# ---------------------------------------------------------------------------
try:
    from TkModules.TkOrderbookAutoencoder import TkOrderbookAutoencoder
    from TkModules.TkLastTradesAutoencoder import TkLastTradesAutoencoder
    ob_ae = TkOrderbookAutoencoder(CFG).eval()
    lt_ae = TkLastTradesAutoencoder(CFG).eval()
    check("B. Автоэнкодеры (VQ-VAE стакан + Дирихле лента) инстанцируются", True)
except Exception as e:
    check("B. Автоэнкодеры инстанцируются", False, repr(e))

# ---------------------------------------------------------------------------
# C. Forecaster: forward на КОРРЕКТНОЙ ширине (InputWidth=20)
# ---------------------------------------------------------------------------
prior = int(CFG["TimeSeries"]["PriorStepsCount"])   # 12
width = int(CFG["TimeSeries"]["InputWidth"])         # 20
code_size = int(CFG["Autoencoders"]["LastTradesAutoencoderCodeLayerSize"])  # 8
forecaster = None
try:
    from TkModules.TkTimeSeriesForecaster import TkTimeSeriesForecaster
    forecaster = TkTimeSeriesForecaster(CFG).eval()
    with torch.no_grad():
        x_ok = torch.randn(1, prior * width)   # 12*20 = 240 — как ждёт модель
        y, y_aux = forecaster(x_ok)
    ok = tuple(y.shape)[-1] == code_size
    check(f"C. forward @ width={width} -> код размера {code_size}", ok, f"выход {tuple(y.shape)}")
except Exception as e:
    check(f"C. forward @ width={width}", False, repr(e))

# ---------------------------------------------------------------------------
# D. БАГ train/serve: инференс даёт 19 признаков/шаг, модель ждёт 20.
#    Подаём вход шириной 19 -> модель должна его ОТВЕРГНУТЬ (reshape падает).
#    PASS здесь = баг подтверждён (serving-код рассинхронизирован с обучением).
# ---------------------------------------------------------------------------
SERVING_WIDTH = 19  # preprocess_samples(): 8+8 + price + ob_vol + lt_vol (без spread)
if forecaster is not None:
    raised = False
    err = ""
    try:
        with torch.no_grad():
            forecaster(torch.randn(1, prior * SERVING_WIDTH))  # 12*19 = 228
    except Exception as e:
        raised = True
        err = type(e).__name__
    check("D. Баг 6.2: вход serving-ширины(19) отвергается моделью(20)", raised,
          f"исключение: {err}" if raised else "НЕ упал — проверь вручную, возможно молчаливое искажение")
    check("D2. Обучение(20) и инференс(19) реально расходятся", width != SERVING_WIDTH,
          f"train={width}, serve={SERVING_WIDTH}")

# ---------------------------------------------------------------------------
# E. Численная корректность TkStatistics на ручных мок-данных
# ---------------------------------------------------------------------------
try:
    from t_tech.invest.schemas import Quotation
    from TkModules.TkStatistics import TkStatistics

    def Q(units, nano=0):
        return Quotation(units=units, nano=nano)

    # Мок стакана: лучший бид 100 (объём 7), аск 101 (объём 5); last_price 100
    ob = SimpleNamespace(
        bids=[SimpleNamespace(price=Q(100), quantity=7)],
        asks=[SimpleNamespace(price=Q(101), quantity=5)],
        last_price=Q(100),
    )
    dist, desc, vol, pivot = TkStatistics.orderbook_distribution(ob, 256, 0.01)
    check("E1. orderbook_distribution: суммарный объём = 12", vol == 12, f"vol={vol}")
    check("E2. orderbook_distribution: pivot = лучший бид (100)", abs(pivot - 100.0) < 1e-6, f"pivot={pivot}")

    # Мок ленты: сделки 100x3 и 101x4 -> объём 7
    trades = SimpleNamespace(trades=[
        SimpleNamespace(price=Q(100), quantity=3),
        SimpleNamespace(price=Q(101), quantity=4),
    ])
    d, ds, v = TkStatistics.trades_distribution(trades, 100.0, 128, 0.01)
    check("E3. trades_distribution: суммарный объём = 7", v == 7, f"vol={v}")
except Exception as e:
    check("E. TkStatistics численные проверки", False, repr(e))

# ---------------------------------------------------------------------------
# Итог
# ---------------------------------------------------------------------------
passed = sum(_results)
total = len(_results)
print("\n" + "=" * 60)
print(f"ИТОГ: {passed}/{total} проверок пройдено")
print("Тест D (баг 6.2): PASS = рассинхрон подтверждён и требует правки serving-кода.")
print("=" * 60)
sys.exit(0 if passed == total else 1)

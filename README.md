# MoexScalp

Deep-learning intraday **scalping bot for the Moscow Exchange (MOEX, TQBR board)**, feeding from the **T-Bank (Tinkoff) Invest API**.

It represents market microstructure — the order book and the trade tape — as **volume-by-price histograms**, compresses them with autoencoders, and forecasts the **distribution of future trades** with an LSTM + attention model. A long signal is raised when the expected upside (the right tail of the forecast distribution) exceeds a threshold.

> **Attribution.** This is a fork of [bad3p/DeepScalp](https://github.com/bad3p/DeepScalp), MIT-licensed. Original copyright © 2025 Alexander Petryaev. See [LICENSE](LICENSE).

---

## Architecture

Three cooperating processes talking over local sockets (`multiprocessing.connection`):

```
TkGatherData ──filename──▶ TkForecastingService ──(ticker,profit)──▶ TkTradingService
 order book +               inference of 3 models                    order execution
 trade tape                 + DearPyGui plots                        (long-only)
 round-robin TQBR
```

**ML pipeline**

- **Order-book autoencoder** — 1D-CNN with a VQ-VAE latent (`VectorQuantizerEMA`), input = cumulative order-book volume distribution (256 bins), output code of size 8.
- **Trade-tape autoencoder** — CNN + residual MLP with a **Dirichlet** latent, input = trade volume distribution (128 bins), softmax output (a proper distribution), code of size 8.
- **Time-series forecaster** — 12 prior steps × 20 features → per-slice LSTMs → multi-head attention fusion (gated residual) → MLP → 8-dim code of the **future** trade distribution, expanded back by the trade-tape decoder.

---

## Prerequisites

- Python 3.10+, **NVIDIA GPU with CUDA 11.8** (the code calls `.cuda()` directly)
- Windows (uses `win10toast`, `.bat` launchers)
- A **T-Bank brokerage account** and an Invest API token

## Installation

```bash
pip install numpy tinkoff-investments dearpygui joblib win10toast jsonpickle
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Access token

Provide the T-Bank Invest API token in the `TK_TOKEN` environment variable on the local machine.

---

## Usage

1. **Gather data** (needs the token; runs during trading hours):
   `TkGatherDataLoop.bat` → writes `./Data/*.obs` snapshots of the order book and trade tape.
2. **Train the autoencoders:**
   `python TkPreprocessAutoencoderData.py` then `python TkTrainAutoencoders.py`.
3. **Train the forecaster:**
   `python TkPreprocessTimeSeriesData.py` then `python TkTrainTimeSeries.py`.
4. **Run live** (in parallel): `TkGatherDataLoop.bat` + `python TkForecastingService.py` (+ `python TkTradingService.py`).

## Testing

`test_deepscalp.py` is an **offline harness (Level 0)** — it needs neither token, GPU, nor collected data (it patches `.cuda()` to run on CPU). Run from the repo root:

```bash
python test_deepscalp.py
```

It checks that all nets build from the config, the autoencoders and forecaster instantiate, the forecaster forward pass works, and the `TkStatistics` volume math is correct — and it empirically flags the train/serve issue below.

---

## Status & known issues

This is a **research prototype**, not a turnkey bot. Before relying on it:

- **No pretrained models are shipped** — `Data/` and `Models/` are empty. You must gather data (days–weeks) and train from scratch.
- **Train/serve feature mismatch.** Training builds **20** features per step (including the spread), while live `preprocess_samples` builds **19** (no spread) and normalizes price differently. The forecaster expects 20 → this must be fixed before inference is meaningful.
- **No backtest.** There is no walk-forward evaluation and no transaction-cost/slippage model, so profitability is unproven. Execution is long-only with no stop-loss.
- **Sampling resolution.** Round-robin polling over all TQBR tickers means each ticker is snapshotted every few minutes, not sub-second — coarse for true scalping.

## License

MIT — see [LICENSE](LICENSE).

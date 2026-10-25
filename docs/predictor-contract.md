# Predictor contract — transformer sidecar (prototype fork)

This document is the wire contract between this App fork and the transformer
sidecar built in the companion training repo. Both sides must implement this
file byte-identically: the tensor builder here (`BuildPredictorTensor`) and
the training corpus builder there encode the same v1 rows, or finetuning
learns a different distribution than serving sees.

Scope: `POST http://predictor:8000/v1/predict` on the shared Docker bridge.
No TLS on that net by design; auth is the shared bearer token, and the
sidecar exposes no published port. If the sidecar ever moves off-host, revisit
with mTLS — explicitly out of scope for the prototype.

## Tensor v1 (`PredictorTensorVersion = "v1"`)

One row per calendar day over the trailing 180 days ending on the owner's
today (`PredictorTensorWindowDays = 180`). Day steps walk through
`AddCalendarDays`; cycle days measure through `CalendarDaysBetween`.

| Field | Meaning |
|-------|---------|
| `date` | `YYYY-MM-DD` calendar day |
| `present` | a log row exists for the day (latest same-day write wins) |
| `bleed` | 1 on a non-spotting period day, else 0 (spotting never reads as bleed, matching the boundary rule) |
| `flow` | 0 none, 1 spotting, 2 light, 3 medium, 4 heavy |
| `bbt_c` | stored Celsius BBT, or null when unmeasured, invalid, or disturbed |
| `bbt_missing` | true unless a valid undisturbed reading backs `bbt_c` |
| `mucus_score` | 0 none, 1 dry, 2 moist, 3 creamy, 4 eggwhite (peak last) |
| `symptom_count` | deduplicated symptom ids on the day |
| `disturbed` | day carries `illness` or `sleep_disruption` (drops the BBT reading entirely) |
| `cycle_day` | 1-based day under the shared `CycleBoundaries`, 0 when before the first boundary |
| `gap_before` | days since the last present day exceed 90 (`PredictorSkipGapDays`): a missed-log merged gap, a skip — never averaged |
| `missing_mask` | bit 0: absent row, bit 1: BBT missing, bit 2: cycle day unknown |

Request envelope: `{user_id, tensor, history_n, recent_lengths}` where
`history_n` is the completed-cycle count (the sidecar decides zero-shot vs
finetuned from it; `n<3` still calls) and `recent_lengths` are the recent
cycle lengths, newest last, over the same recent window the median uses.

## Tensor v2 (`PredictorTensorVersionV2 = "v2"`)

V2 is the same 180-day window with two extra day fields feeding the
mixed-v2 weights (`artifacts/mixed_v2/model.onnx`, 14 columns):

| Field | Meaning |
|-------|---------|
| `peak` | 1 on an observed libido-peak day, else 0 |
| `peak_missing` | 1 when the peak value is unknown for the day, else 0 |

Tracking-aware mapping (window = the same 180d `byDay` map; `active` = any
day in-window with `LibidoPeak` set):

- present + checked → `peak=1, peak_missing=0` (observed peak).
- present + unchecked + active tracker → `peak=0, peak_missing=0`
  (known-absent: the owner tracks, so an unchecked box means no peak).
- present + unchecked + inactive tracker → `peak=0, peak_missing=1`
  (unknown: an unchecked box from a never-tracking owner is not
  known-absent; emitting known-absent everywhere would mimic anovulatory
  Fehring cycles, a regime the model never saw on ovulatory windows).
- absent → `peak=0, peak_missing=1` (same as Python `build_tensor_v2`).

Version routing: the sidecar accepts `tensor.version` in `{"v1", "v2"}`
(still 180 days, else 422). The v1 builder is untouched and v1 stacks keep
sending v1; the client builds v2 only when `PredictorConfig.TensorVersion`
is `"v2"` (`PREDICTOR_TENSOR_VERSION=v2`, set automatically by
`make up PREDICTOR=v2`). The reshape width follows the loaded model, so a
version/model mismatch fails soft into the median fallback.

Calibration: the bundled `normalizer.json` carries `interval_widen_days`
(v2 ships 3); the sidecar widens the decoded p10/p90 by k days each before
replying. p50 is untouched. The v1 normalizer has no such key and reads as
0 (no widening).

Model selection: `Menstral/Dockerfile.cpu` takes `MODEL_DIR` (default
`artifacts` for v1; `artifacts/mixed_v2` via `make build-v2`). The v2 stack
advertises `MODEL_VERSION=ovumcy-mixed-v2`, which surfaces in stats via the
existing `model_version` field. No production tag in this plan: v2 serves
as a staged trial stack only.

## Response

Success: `{next_onset, ovulation_date, fertile_start, fertile_end,
p10, p50, p90, model_version}` — dates as `YYYY-MM-DD`. Either date may be
empty when that half has no answer. Abstention: `{no_prediction: true,
reason}` — the app keeps the legacy median and renders existing empty
states. Decoding is strict: unknown fields are refused and the call falls
back to legacy.

## App-side rules

- Gates run before any network I/O (`PredictionsSuppressed` skips the call
  entirely; the fertility half overlays only where
  `FertilityProjectionSuppressed` is open). Never re-derive per surface: all
  suppression stays in `dashboard_cycle.go`.
- One retry on 5xx only; timeouts/invalid bodies fall back with no retry.
  Default per-attempt timeout 800ms (`PREDICTOR_TIMEOUT_MS`).
- Cache: in-memory read-through on `(user_id, input_hash)`, 24h TTL.
- Narrow neural windows widen to at least the legacy width before display.
- Logs carry `user hash, input_hash, model_version, latency` only — never
  raw BBT, never the raw user id.
- Env: `PREDICTOR_ENABLED` (default false), `PREDICTOR_URL`
  (default `http://predictor:8000`), `PREDICTOR_TOKEN` (or
  `PREDICTOR_TOKEN_FILE`), `PREDICTOR_TIMEOUT_MS` (default 800),
  `PREDICTOR_TENSOR_VERSION` (default v1; exactly `v2` routes the mixed-v2 trial).

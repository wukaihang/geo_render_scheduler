# Innovation One Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, hardware-decoupled implementation of content- and runtime-aware per-GPU duration prediction and expected-finish-time scheduling for innovation one.

**Architecture:** Stable dataclasses and protocols isolate request, device, result, and hardware contracts. Prediction and scheduling remain pure/testable, while a deterministic discrete-event replay provides the complete experimental loop without claiming simulated values are RTX 5090 measurements.

**Tech Stack:** Python 3.9+, NumPy, scikit-learn, joblib, pytest, standard-library argparse/CSV/JSON.

## Global Constraints

- Innovation one assumes every model is already resident on every GPU; no load, eviction, replication, or cache-aware score may appear.
- P95 render predictions drive EFT and queued-work estimates; P50 is retained for prediction evaluation.
- Only completed requests may update online history.
- Train/validation/test splitting is by complete session/trajectory group.
- Synthetic results must be labeled `source=synthetic`; hardware backends must fail explicitly until implemented on Linux EGL/NVML hardware.
- Every policy receives the same immutable trace and stable GPU ordering.
- Unknown telemetry is `None`, never a fabricated zero.

---

### Task 1: Package skeleton, domain types, and hardware boundary

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/geo_render/__init__.py`
- Create: `src/geo_render/common/__init__.py`
- Create: `src/geo_render/common/errors.py`
- Create: `src/geo_render/common/types.py`
- Create: `src/geo_render/rendering/__init__.py`
- Create: `src/geo_render/rendering/interface.py`
- Create: `src/geo_render/rendering/unavailable.py`
- Test: `tests/unit/test_types.py`
- Test: `tests/unit/test_hardware_boundary.py`

**Interfaces:**
- Produces: immutable `Camera`, `RenderRequest`, `ModelManifest`, `DeviceState`, `DurationPrediction`, `QueuedRequest`, `WorkerSnapshot`, `CostEstimate`, `ScheduleDecision`, `RenderResult`, and `TraceRecord` dataclasses.
- Produces: `Renderer.render(request, gpu_id) -> RenderResult` and `DeviceStateProvider.snapshot() -> tuple[DeviceState, ...]` protocols.
- Produces: `HardwareBackendUnavailable` and unavailable backends that always raise it with actionable platform requirements.

- [x] **Step 1: Write failing validation and hardware-boundary tests**

```python
def test_request_rejects_non_positive_dimensions(sample_request):
    with pytest.raises(ValidationError, match="output_width"):
        replace(sample_request, output_width=0)

def test_unavailable_renderer_never_fabricates_measurements(sample_request):
    with pytest.raises(HardwareBackendUnavailable, match="Linux.*EGL.*VTK"):
        UnavailableRenderer().render(sample_request, "gpu-0")
```

- [x] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_types.py tests/unit/test_hardware_boundary.py -q`

Expected: collection fails because `geo_render` does not exist.

- [x] **Step 3: Add packaging, validated dataclasses, protocols, and unavailable backends**

Use frozen dataclasses with `__post_init__` validation. `TraceRecord.actual_render_ms_by_gpu` is immutable and is not part of `RenderRequest`, preventing online policies from observing oracle values through the request API.

```python
class Renderer(Protocol):
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult: ...

class UnavailableRenderer:
    def render(self, request: RenderRequest, gpu_id: str) -> RenderResult:
        raise HardwareBackendUnavailable(
            "Real rendering requires Linux, NVIDIA drivers, VTK EGL, and a "
            "verified per-process GPU binding adapter."
        )
```

- [x] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_types.py tests/unit/test_hardware_boundary.py -q`

Expected: all Task 1 tests pass.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/geo_render/common src/geo_render/rendering tests/unit/test_types.py tests/unit/test_hardware_boundary.py
git commit -m "feat: add innovation one domain contracts"
```

### Task 2: Feature extraction, online history, and leakage-safe splitting

**Files:**
- Create: `src/geo_render/prediction/__init__.py`
- Create: `src/geo_render/prediction/features.py`
- Create: `src/geo_render/prediction/online_stats.py`
- Create: `src/geo_render/prediction/dataset.py`
- Test: `tests/unit/test_features.py`
- Test: `tests/unit/test_online_stats.py`
- Test: `tests/unit/test_dataset.py`

**Interfaces:**
- Consumes: domain types from Task 1.
- Produces: `extract_features(request, manifest, device, history) -> dict[str, float | str]`.
- Produces: `OnlineDurationStats(alpha, window_size, default_ms)` with `observe(...)` and hierarchical `estimate(...)`.
- Produces: `group_split(records, train_fraction, validation_fraction, seed)` returning three disjoint record tuples.

- [x] **Step 1: Write failing geometry, update-order, and leakage tests**

```python
def test_feature_extraction_normalizes_view_direction(sample_inputs):
    values = extract_features(*sample_inputs)
    norm = sum(values[key] ** 2 for key in ("view_x", "view_y", "view_z")) ** 0.5
    assert norm == pytest.approx(1.0)

def test_online_history_changes_only_after_observe(stats):
    before = stats.estimate("gpu-0", "model-a")
    stats.observe("gpu-0", "model-a", 20.0)
    assert before.p50_ms != stats.estimate("gpu-0", "model-a").p50_ms

def test_group_split_never_splits_a_trajectory(records):
    train, validation, test = group_split(records, 0.6, 0.2, seed=7)
    groups = [{r.request.trajectory_id for r in part} for part in (train, validation, test)]
    assert groups[0].isdisjoint(groups[1] | groups[2])
    assert groups[1].isdisjoint(groups[2])
```

- [x] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_features.py tests/unit/test_online_stats.py tests/unit/test_dataset.py -q`

Expected: imports fail because feature/history modules are missing.

- [x] **Step 3: Implement pure features, completed-only online statistics, and group splitting**

The history fallback order is `(gpu, model) -> gpu -> global -> configured default`. EWMA uses `alpha`; empirical P95 uses a bounded deque. Group assignment uses a seeded shuffle of group IDs, never a row shuffle.

- [x] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_features.py tests/unit/test_online_stats.py tests/unit/test_dataset.py -q`

Expected: all Task 2 tests pass.

- [x] **Step 5: Commit**

```bash
git add src/geo_render/prediction tests/unit/test_features.py tests/unit/test_online_stats.py tests/unit/test_dataset.py
git commit -m "feat: extract render features and online history"
```

### Task 3: Duration predictors, evaluation, and artifact persistence

**Files:**
- Create: `src/geo_render/prediction/interface.py`
- Create: `src/geo_render/prediction/baselines.py`
- Create: `src/geo_render/prediction/models.py`
- Create: `src/geo_render/prediction/artifacts.py`
- Create: `src/geo_render/analysis/__init__.py`
- Create: `src/geo_render/analysis/prediction_metrics.py`
- Test: `tests/unit/test_predictors.py`
- Test: `tests/unit/test_prediction_metrics.py`

**Interfaces:**
- Consumes: feature dictionaries and completed render labels.
- Produces: `DurationPredictor.predict(features) -> DurationPrediction`.
- Produces: `GlobalMeanPredictor`, `EWMAPredictor`, `RidgeDurationPredictor`, and `QuantileGBDTPredictor`.
- Produces: `save_artifact(...)`, `load_artifact(...)`, and `prediction_metrics(actual, p50, p95)`.

- [ ] **Step 1: Write failing predictor contract, persistence, and metric tests**

```python
@pytest.mark.parametrize("factory", predictor_factories())
def test_predictors_return_ordered_positive_quantiles(factory, training_rows):
    predictor = factory().fit(training_rows)
    prediction = predictor.predict(training_rows[0].features)
    assert 0 < prediction.p50_ms <= prediction.p95_ms

def test_artifact_round_trip_preserves_predictions(tmp_path, fitted_predictor, row):
    path = tmp_path / "model.joblib"
    save_artifact(path, fitted_predictor, training_sha256="abc")
    restored = load_artifact(path)
    assert restored.predict(row.features) == fitted_predictor.predict(row.features)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_predictors.py tests/unit/test_prediction_metrics.py -q`

Expected: imports fail because predictor modules are missing.

- [ ] **Step 3: Implement baseline and scikit-learn predictors**

Use `DictVectorizer(sparse=False)` plus `Ridge` for the linear model. Use two `GradientBoostingRegressor(loss="quantile", alpha=0.5/0.95)` instances for GBDT. Clamp predictions to a configured positive floor and enforce `p95=max(p50,p95)`. Persist a versioned envelope containing feature schema version, training hash, library versions, and model.

- [ ] **Step 4: Implement prediction metrics**

```python
return {
    "mae_ms": mean(abs(actual - p50)),
    "mape": mean(abs(actual - p50) / maximum(actual, epsilon)),
    "p95_absolute_error_ms": percentile(abs(actual - p50), 95),
    "underestimate_rate": mean(p50 < actual),
    "p95_coverage": mean(actual <= p95),
}
```

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_predictors.py tests/unit/test_prediction_metrics.py -q`

Expected: all Task 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/geo_render/prediction src/geo_render/analysis tests/unit/test_predictors.py tests/unit/test_prediction_metrics.py
git commit -m "feat: add duration prediction models"
```

### Task 4: Worker state machine and six scheduling policies

**Files:**
- Create: `src/geo_render/scheduling/__init__.py`
- Create: `src/geo_render/scheduling/interface.py`
- Create: `src/geo_render/scheduling/state.py`
- Create: `src/geo_render/scheduling/policies.py`
- Test: `tests/unit/test_scheduler_state.py`
- Test: `tests/unit/test_policies.py`

**Interfaces:**
- Consumes: requests, devices, model manifests, and duration predictors.
- Produces: `SchedulerContext`, `SchedulerPolicy`, `WorkerState`, `ClusterState`.
- Produces: `RoundRobinPolicy`, `LeastQueuePolicy`, `StaticWeightedPolicy`, `EFTPolicy` configured as EWMA or Feature, and `OracleEFTPolicy.for_offline_replay(...)`.

- [ ] **Step 1: Write failing state transition and policy decision tests**

```python
def test_feature_eft_selects_lowest_expected_completion(context, request):
    decision = EFTPolicy(feature_predictor).choose(request, context)
    assert decision.gpu_id == "gpu-1"
    assert decision.costs["gpu-1"].total_ms < decision.costs["gpu-0"].total_ms

def test_oracle_cannot_be_constructed_for_online_use():
    with pytest.raises(OracleAccessError):
        OracleEFTPolicy()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_scheduler_state.py tests/unit/test_policies.py -q`

Expected: imports fail because scheduling modules are missing.

- [ ] **Step 3: Implement state invariants and deterministic policies**

Store queued P95 at assignment. Compute running remainder from predicted finish and `now_ms`. Use stable GPU ID tie-breaking. The online request object never exposes `actual_render_ms_by_gpu`; only the offline oracle receives the trace record through a private mapping.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_scheduler_state.py tests/unit/test_policies.py -q`

Expected: all Task 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/geo_render/scheduling tests/unit/test_scheduler_state.py tests/unit/test_policies.py
git commit -m "feat: implement EFT and baseline schedulers"
```

### Task 5: Deterministic trace generation and discrete-event replay

**Files:**
- Create: `src/geo_render/workload/__init__.py`
- Create: `src/geo_render/workload/trace.py`
- Create: `src/geo_render/workload/synthetic.py`
- Create: `src/geo_render/workload/replay.py`
- Test: `tests/unit/test_trace.py`
- Test: `tests/integration/test_replay.py`

**Interfaces:**
- Consumes: manifests, devices, policies, and frozen `TraceRecord` sequences.
- Produces: `generate_synthetic_trace(config)`, `read_trace_csv`, `write_trace_csv`, and `ReplayEngine.run(trace, policy)`.
- Produces: immutable completed-request and decision-event records.

- [ ] **Step 1: Write failing determinism, fairness, and future-isolation tests**

```python
def test_trace_is_identical_for_same_seed(config):
    assert generate_synthetic_trace(config) == generate_synthetic_trace(config)

def test_every_policy_completes_the_same_request_set(trace, policies):
    completed = [ReplayEngine(devices).run(trace, policy).completed for policy in policies]
    assert all({r.request_id for r in rows} == {t.request.request_id for t in trace} for rows in completed)

def test_online_policy_never_receives_trace_record(trace, spying_policy):
    ReplayEngine(devices).run(trace, spying_policy)
    assert spying_policy.observed_types == {RenderRequest}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_trace.py tests/integration/test_replay.py -q`

Expected: imports fail because workload modules are missing.

- [ ] **Step 3: Implement seeded Poisson generation and event replay**

Use `random.Random(seed).expovariate(arrival_rate_per_ms)`. Heap events are ordered by `(time_ms, completion_before_arrival, sequence)`. On completion, update only the selected online predictor/history observer, then start that GPU's queue head. `source="synthetic"` is mandatory on generated traces and replay metadata.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_trace.py tests/integration/test_replay.py -q`

Expected: all Task 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/geo_render/workload tests/unit/test_trace.py tests/integration/test_replay.py
git commit -m "feat: add deterministic workload replay"
```

### Task 6: Scheduling metrics and self-contained experiment outputs

**Files:**
- Create: `src/geo_render/analysis/scheduling_metrics.py`
- Create: `src/geo_render/experiments/__init__.py`
- Create: `src/geo_render/experiments/io.py`
- Create: `src/geo_render/experiments/compare.py`
- Test: `tests/unit/test_scheduling_metrics.py`
- Test: `tests/integration/test_experiment_outputs.py`

**Interfaces:**
- Consumes: replay result and experiment configuration.
- Produces: `scheduling_metrics(result) -> dict[str, object]`.
- Produces: `write_run_directory(...)` and `compare_policies(...)`.

- [ ] **Step 1: Write failing metric and artifact-completeness tests**

```python
def test_run_directory_is_self_contained(tmp_path, small_experiment):
    output = write_run_directory(tmp_path / "run", small_experiment)
    assert {p.name for p in output.iterdir()} == {
        "config.json", "trace.csv", "requests.csv", "decisions.jsonl",
        "summary.json", "metadata.json",
    }

def test_jain_index_is_one_for_equal_user_slowdown():
    assert jain_index([2.0, 2.0]) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/unit/test_scheduling_metrics.py tests/integration/test_experiment_outputs.py -q`

Expected: imports fail because metrics/experiment modules are missing.

- [ ] **Step 3: Implement latency, SLO, utilization, fairness, and workload-balance metrics**

Percentiles use NumPy's linear method. Slowdown uses each trace record's isolated P50 label when present; missing labels produce `null` slowdown metrics rather than fabricated values. Throughput divides completed requests by the observed arrival-to-finish window.

- [ ] **Step 4: Implement atomic run-directory writing and policy comparison**

Write into a sibling temporary directory, flush every file, then rename to the final path. Existing final directories are rejected. `metadata.json` includes `source`, seed, trace SHA-256, Python/library versions, and Git commit if available.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python3 -m pytest tests/unit/test_scheduling_metrics.py tests/integration/test_experiment_outputs.py -q`

Expected: all Task 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/geo_render/analysis src/geo_render/experiments tests/unit/test_scheduling_metrics.py tests/integration/test_experiment_outputs.py
git commit -m "feat: report innovation one experiment metrics"
```

### Task 7: CLI, sample configuration, documentation, and end-to-end verification

**Files:**
- Create: `src/geo_render/cli.py`
- Create: `src/geo_render/__main__.py`
- Create: `configs/synthetic_example.json`
- Modify: `README.md`
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: all earlier public APIs.
- Produces: `geo-render generate-trace`, `geo-render train`, `geo-render compare`, and `geo-render check-hardware` commands.

- [ ] **Step 1: Write failing CLI smoke tests**

```python
def test_compare_cli_creates_policy_summaries(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "geo_render", "compare", "--config", config,
         "--output", str(tmp_path / "results")],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results" / "comparison.json").exists()

def test_hardware_check_fails_actionably():
    result = run_cli("check-hardware")
    assert result.returncode != 0
    assert "Linux" in result.stderr and "EGL" in result.stderr
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/integration/test_cli.py -q`

Expected: command module is missing.

- [ ] **Step 3: Implement argparse commands and example config**

`generate-trace` creates and hashes a frozen CSV. `train` group-splits profile CSV and persists all requested predictor artifacts plus metrics. `compare` runs the six policies on one trace and writes a top-level comparison. `check-hardware` invokes unavailable adapters and exits with code 2 until the real adapter exists.

- [ ] **Step 4: Document installation, commands, data provenance, and hardware extension points**

README must distinguish synthetic verification from paper evidence, show exact commands, define input/output schemas, and name `rendering/interface.py` as the hardware integration seam.

- [ ] **Step 5: Run focused and full verification**

Run: `python3 -m pytest tests/integration/test_cli.py -q`

Expected: CLI tests pass.

Run: `python3 -m pytest -q`

Expected: all tests pass with no warnings produced by project code.

Run: `python3 -m geo_render compare --config configs/synthetic_example.json --output build/example-results`

Expected: exit code 0 and `build/example-results/comparison.json` lists all six policies.

Run: `python3 -m geo_render check-hardware`

Expected: exit code 2 and an actionable Linux/NVIDIA/VTK EGL message.

- [ ] **Step 6: Audit design coverage and repository state**

Run: `rg -n "TODO|TBD|NotImplementedError|pass$" src tests README.md configs || true`

Expected: no unresolved implementation markers; protocol method ellipses are permitted only in `rendering/interface.py` and `prediction/interface.py`.

Run: `git status --short`

Expected: only intentional final documentation or verification artifacts are uncommitted.

- [ ] **Step 7: Commit**

```bash
git add src/geo_render/cli.py src/geo_render/__main__.py configs/synthetic_example.json README.md tests/integration/test_cli.py
git commit -m "feat: complete innovation one experiment workflow"
```

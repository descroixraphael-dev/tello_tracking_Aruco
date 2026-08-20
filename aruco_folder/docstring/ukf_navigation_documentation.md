# `ukf_navigation.py` (`orbit_nav.py` / `TelloNavigationController`) — Documentation

This doc only covers what **differs** from `single_axis.py`. Shared pieces —
`quaternion_to_rotation_matrix`, the `UKF_CTRV` math (`predict`/`update`/
`reinitialise`), `PIDController.compute()`'s actual logic, `ALTITUDE_SETTLE_*`
constants, `LOSS_*` constants, and the general `ALTITUDE_CALIBRATE` →
`AXIS_TEST` flight-state shape — are identical and already covered in
`single_axis_documentation.md`.

---

## 1. Role: this is the real controller, not a bench

`single_axis.py` isolates one axis at a time (commented-out blocks). This
file's `_run_axis_test` **always drives two axes simultaneously, every
tick** — there's no axis-selection block:

```python
twist.linear.x = self.pid_z.compute(err_z, dt)                       # forward hold
twist.angular.z = self.pid_yaw.compute(atan2(marker_err_x, marker_err_z), dt)  # yaw-to-marker
```

No `pid_x` (lateral) or altitude re-drive happens in this phase — altitude
is still handled once, up front, in `ALTITUDE_CALIBRATE`, same principle as
`single_axis.py`. The `'AXIS_TEST'` state name is kept purely for parity
with `single_axis.py`'s FSM naming, per the class docstring — it isn't a
test bench here.

---

## 2. Module-level constant differences

| Name | `single_axis.py` | `ukf_navigation.py` |
|---|---|---|
| `OFFSET_TOWARD_M` | `1.0` | `1.50` |
| `OFFSET_NORMAL_M` | `0.35` | `0.35` *(same)* |

⚠️ **Stale comments, not code bugs, but worth fixing:** the constants block
comment still says *"the drone holds 1m behind / 35cm above the marker"* —
that "1m" doesn't match the actual `OFFSET_TOWARD_M = 1.50`. Likely left
over from before the toward-offset was retuned. Similarly, the comment
inside `_run_axis_test` says *"the 1m-behind/45cm-above offset was already
applied upstream"* — both the "1m" and "45cm" are stale (actual values are
1.5 m / 0.35 m).

No `current_zone` / zone parsing here — `_pose_callback`'s `frame_id`
parsing assumes a bare `"<marker_id>"` (still splits on `":"` defensively,
but nothing reads a zone component). `single_axis.py` parses and stores
`self.current_zone` from a `"<id>:<zone>"` format, even though it's
currently informational-only there too.

---

## 3. No debug telemetry publishers

`single_axis.py` publishes `/tello/nav/velocity_estimate` and
`/tello/nav/yaw_rate_estimate` for PlotJuggler (`_init_debug_telemetry`,
`_publish_velocity_estimate`, `_publish_yaw_rate_estimate`). None of that
exists in `ukf_navigation.py` — no `Vector3Stamped` import, no debug
publishers, no calls to publish UKF speed/heading/turn-rate anywhere.

---

## 4. `UKF_CTRV` — same math, different packaging

Functionally identical (`reinitialise`, `predict`, `update` are the same
logic, just with shorter local variable names — `d`/`zd`/`xd` instead of
`diff`/`z_diff`/`x_diff`). Structural differences only:

- `_Q_DIAG`, `_R_DIAG`, `_ALPHA`, `_KAPPA`, `_BETA` are **class-level**
  constants here, vs. being set directly as instance attributes inside
  `__init__` in `single_axis.py`. Values are identical
  (`R = diag([0.002]*3)` in both).
- `self.P` initialized as `np.eye(n)` here vs. `np.eye(n) * 1.0` in
  `single_axis.py` — numerically identical, just written differently.
- `self.sigma_pred: np.ndarray | None = None` is explicitly declared in
  `__init__` here; `single_axis.py` doesn't pre-declare it (it only exists
  once `predict()` runs first). No behavioral difference, just an extra
  type-annotated placeholder.

---

## 5. `PIDController` — docstring doesn't match the code (in both files, but only documented here)

This file's class docstring says:
```
- Integral anti-windup (clamp to ±max_out / ki)
- Sign-change reset to prevent integral wind-up after overshoot
```
But `compute()` is byte-for-byte the same as `single_axis.py`'s — **only**
sign-change reset exists; there's no clamp-based integral undo anywhere in
either file. This is a stale/inaccurate docstring specific to this file
(`single_axis.py`'s `PIDController` has no such docstring claim to begin
with), not an actual behavioral difference between the two files.

### PID gain differences

| PID | Param | `single_axis.py` | `ukf_navigation.py` |
|---|---|---|---|
| `pid_z` | `min_effective_out` | `0.10` | `0.15` |
| `pid_yaw` | `max_out` | `0.50` | `0.90` |

Everything else (`kp`/`ki`/`kd` for all four PIDs, and `pid_x`/`pid_y`'s
`max_out`/`min_effective_out`) matches exactly. The `pid_yaw.max_out`
difference is the more consequential one — this file allows nearly double
the yaw authority (`0.90` vs `0.50`) that the bench does.

---

## 6. New: `self.last_yaw_error` — bearing-trend tracker (not in `single_axis.py` at all)

A persistent tracker with no equivalent in `single_axis.py`:
- Initialized to `0.0` in `__init__`.
- Updated **every tick** in `_run_axis_test`, right after computing
  `yaw_error = atan2(marker_err_x, marker_err_z)`.
- **Re-baselined on re-acquisition** in `_pose_callback` (set to
  `atan2(marker_pos[0], marker_pos[2])`) so it doesn't look like a sudden
  jump right after re-lock.

Its only consumer is `_handle_tracking_loss`'s `SEARCHING` branch:
```python
search_dir = 1.0 if self.last_yaw_error >= 0.0 else -1.0
search_twist.angular.z = search_dir * SEARCH_YAW_RATE
```
This makes the search-scan continue in whichever direction the marker was
last drifting off-frame, rather than always rotating the same way.
`single_axis.py`'s `_handle_tracking_loss` has no such tracker — its
`SEARCHING` branch always scans at a fixed `+SEARCH_YAW_RATE`, regardless of
which way the marker actually left the frame.

---

## 7. Structural/naming differences

- Methods are private-prefixed here (`_pose_callback`, `_control_loop`,
  `_trigger_takeoff`) vs. public names in `single_axis.py`
  (`pose_callback`, `control_loop`, `trigger_takeoff`).
- `_reset_and_stop()` — a small helper here (reset all PIDs + publish zero
  `Twist`), used in `_control_loop`'s "no measurement yet" early return.
  `single_axis.py` does the same two steps inline, without a named helper.
- Node name: `'tello_nav_controller'` vs. `'tello_single_axis_test'`.
- Class name: `TelloNavigationController` vs. `SingleAxisTestController`.

---

## 8. Docstring claims a marker-heading rotation that doesn't appear in the code

The module and class docstrings both claim position errors are **"rotated
into the marker's own heading frame"** using the UKF's heading state so a
rotating marker doesn't cause lateral drift. Looking at `_run_axis_test`,
no such rotation actually happens — `err_z` and `yaw_error` are used
directly, with no additional rotation step by marker heading or UKF θ. This
looks like an aspirational or stale docstring rather than a description of
the current implementation; worth confirming with whoever wrote it before
relying on that claim, since `single_axis.py` makes no equivalent claim and
neither file's code currently does this.

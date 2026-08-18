# AmazingHandGUI — Requirements & Acceptance Criteria

This document captures the functional requirements and acceptance criteria
derived from the current implementation. Each requirement references the
source file(s) where the behaviour is implemented.

---

## 1. Connection Management

### FR-CONN-1: Serial Port Selection
The GUI provides a combo box listing auto-detected serial ports.

| AC | Criterion |
|----|-----------|
| 1.1 | On Linux, `/dev/ttyACM*`, `/dev/ttyUSB*`, `/dev/ttyAMA*` devices appear; if none exist, `/dev/ttyACM0` and `/dev/ttyUSB0` are listed as fallbacks. |
| 1.2 | On Windows, `COM1`–`COM20` are listed. |
| 1.3 | The default matches the platform-specific value from `config.yaml` (`/dev/ttyACM0` or `COM9`). |

### FR-CONN-2: Baudrate Selection
A combo box offers configurable baudrate options.

| AC | Criterion |
|----|-----------|
| 2.1 | Options come from `config.yaml` → `serial.baudrate_options` (default `[9600, 115200, 1000000]`). |
| 2.2 | Default selection is `1000000`. |
| 2.3 | The dropdown is disabled while connected. |

### FR-CONN-3: Connect / Disconnect
Connect and Disconnect buttons manage the serial connection and servo torque.

| AC | Criterion |
|----|-----------|
| 3.1 | Connect opens the serial port, enables torque on servos 1–8, disables Connect/port/baud controls, enables Disconnect. |
| 3.2 | Disconnect disables torque on all 8 servos, re-enables Connect/port/baud, disables Disconnect. |
| 3.3 | A connection failure shows an error in the status bar and log; GUI stays disconnected. |

### FR-CONN-4: Auto-Connect on Startup
The GUI attempts to connect automatically 100 ms after launch.

| AC | Criterion |
|----|-----------|
| 4.1 | `connect_controller()` is called via `root.after(100, …)` during init. |

### FR-CONN-5: CLI Connection
The CLI connects via `--port` and `--baudrate` arguments.

| AC | Criterion |
|----|-----------|
| 5.1 | `--port` and `--baudrate` override defaults. |
| 5.2 | Torque is enabled on all 8 servos on connect. |
| 5.3 | Torque is disabled on exit (including Ctrl+C via `finally` block). |
| 5.4 | `--list` does **not** open a hardware connection. |

---

## 2. Finger Control

### FR-FING-1: Four Finger Widgets
Four finger controls are displayed: Ring, Middle, Pointer, Thumb — each with 2 servos.

| AC | Criterion |
|----|-----------|
| 1.1 | Exactly 4 `FingerControl` widgets render with names matching `config.yaml` servo pairs: Ring (5,6), Middle (3,4), Pointer (1,2), Thumb (7,8). |

### FR-FING-2: Auto Mode (Base + Side)
Auto mode provides a vertical close/open slider and a horizontal side-to-side slider.

| AC | Criterion |
|----|-----------|
| 2.1 | Vertical slider: 0° (open) to 110° (closed); top = closed, bottom = open. |
| 2.2 | Horizontal slider: −40° to +40°. |
| 2.3 | Moving either slider sends interpolated positions (via `compute_auto_positions`) to both servos. |

### FR-FING-3: Raw Mode
Raw mode shows two independent vertical sliders (one per servo).

| AC | Criterion |
|----|-----------|
| 3.1 | Selecting Raw hides Auto sliders and shows two vertical sliders per servo (−40 to 110). |
| 3.2 | Mimic checkbox is disabled and unchecked; Center button is disabled. |
| 3.3 | Switching modes syncs values bidirectionally (auto ↔ raw via `decompose_servo_positions`). |

### FR-FING-4: Speed Control
Each finger has a speed combo box (1–6).

| AC | Criterion |
|----|-----------|
| 4.1 | Range is `speeds.min` (1) to `speeds.max` (6), default `speeds.default` (3). |
| 4.2 | Speed is sent via `write_goal_speed()` per servo before position commands. |

### FR-FING-5: Mimic Mode
Close/open changes on a mimicking finger are propagated to all other fingers with Mimic enabled.

| AC | Criterion |
|----|-----------|
| 5.1 | Enabling Mimic on A and B causes A's close/open slider changes to be reflected on B and vice-versa. |
| 5.2 | Mimic only applies in Auto mode; switching to Raw disables it. |

### FR-FING-6: Center Button
Resets the side offset to 0°.

| AC | Criterion |
|----|-----------|
| 6.1 | Clicking Center sets `side_var` to 0 and triggers a position update. |
| 6.2 | Center is disabled in Raw mode. |

### FR-FING-7: Mouse Wheel on Position Slider
Scroll wheel adjusts position ±5°.

| AC | Criterion |
|----|-----------|
| 7.1 | Scroll up → +5° (close), scroll down → −5° (open), clamped to limits. |

### FR-FING-8: LED Activity Indicator
Each finger shows a status LED.

| AC | Criterion |
|----|-----------|
| 8.1 | Moving (moving flag = true) → blinking green at ~350 ms interval. |
| 8.2 | Blocked (goal-vs-position error ≥ 8° and not moving) → solid red. |
| 8.3 | Idle → gray. |

---

## 3. Keyboard Control

### FR-KEY-1: Finger Selection
Keys 1–4 select the active finger.

| AC | Criterion |
|----|-----------|
| 1.1 | 1 = Ring, 2 = Middle, 3 = Pointer, 4 = Thumb. |
| 1.2 | Status bar shows the selected finger name. |

### FR-KEY-2: Arrow Key Movement
Arrow keys move the selected finger.

| AC | Criterion |
|----|-----------|
| 2.1 | Up = close (increase position), Down = open (decrease). |
| 2.2 | Right = increase side offset, Left = decrease. |

### FR-KEY-3: Precision Modifiers
Step size varies by modifier key.

| AC | Criterion |
|----|-----------|
| 3.1 | No modifier: 1° (precise). |
| 3.2 | Shift: 5° (normal). |
| 3.3 | Ctrl: 10° (fast). |
| 3.4 | Status bar shows mode name and resulting angle. |

### FR-KEY-4: Quick Actions
Single-key shortcuts for common actions.

| AC | Criterion |
|----|-----------|
| 4.1 | Q = fully close to 110°. |
| 4.2 | E = fully open to 0°. |
| 4.3 | C = center side to 0°. |

---

## 4. Global Controls

### FR-GLOB-1: Open All
Sets all fingers to fully open.

| AC | Criterion |
|----|-----------|
| 1.1 | All `pos_var` → 0, all `side_var` → 0, positions sent to hardware. |

### FR-GLOB-2: Close All
Sets all fingers to fully closed.

| AC | Criterion |
|----|-----------|
| 2.1 | All `pos_var` → 110, all `side_var` → 0, positions sent. |

### FR-GLOB-3: Center All
Resets all side offsets.

| AC | Criterion |
|----|-----------|
| 3.1 | All `side_var` → 0, positions sent. |

### FR-GLOB-4: Global Speed
A dropdown sets all finger speeds at once.

| AC | Criterion |
|----|-----------|
| 4.1 | Selecting a value updates every per-finger speed combo box. |
| 4.2 | Speed clamped to [1, 6]. |

---

## 5. Pose Management

### FR-POSE-1: Save Pose
User enters a name and saves the current 8-servo positions.

| AC | Criterion |
|----|-----------|
| 1.1 | Positions from all 4 fingers (8 values) are captured via `get_positions()`. |
| 1.2 | Name is validated via `validate_name()` before saving. |
| 1.3 | On success: dropdown refreshes (sorted), entry clears, status bar confirms. |
| 1.4 | Invalid or empty name shows an error messagebox. |

### FR-POSE-2: Apply Pose
Selecting a pose and clicking Apply moves the hand to that pose.

| AC | Criterion |
|----|-----------|
| 2.1 | The 8 positions are applied to all finger widgets. |
| 2.2 | Servo positions are sent to hardware. |
| 2.3 | Delay is estimated from movement distance and speed; pose completion is logged after that delay with target vs. actual comparison. |

### FR-POSE-3: Delete Pose
Removes the selected pose after confirmation.

| AC | Criterion |
|----|-----------|
| 3.1 | A yes/no messagebox asks for confirmation. |
| 3.2 | On confirm: pose removed from config, YAML saved, dropdown refreshed. |
| 3.3 | If no poses remain, dropdown shows `<no poses>`. |

### FR-POSE-4: Name Validation
Names are validated to prevent YAML corruption.

| AC | Criterion |
|----|-----------|
| 4.1 | Empty / whitespace-only → rejected. |
| 4.2 | Longer than 50 characters → rejected. |
| 4.3 | Contains `: { } [ ] , & * # ? \| - < > = ! % @ \` " '` → rejected. |
| 4.4 | Contains control characters (ASCII < 32) → rejected. |
| 4.5 | Leading/trailing spaces → rejected. |

---

## 6. Sequence Management

### FR-SEQ-1: Sequence Player (Main Window)
Dropdown selection, Loop checkbox, Play / Pause / Stop buttons.

| AC | Criterion |
|----|-----------|
| 1.1 | Dropdown lists all saved sequences (or `<no sequences>`). |
| 1.2 | Loop checkbox enables continuous repeat. |
| 1.3 | Play starts sequence in a background thread. |
| 1.4 | Pause toggles paused/resumed; button text switches between "⏸ Pause" and "▶ Resume". |
| 1.5 | Stop sets `stop_sequence = True`; sequence thread terminates. |

### FR-SEQ-2: Sequence Execution Engine
Sequences run in a background thread with interruptible sleeps.

| AC | Criterion |
|----|-----------|
| 2.1 | Pose steps parse `"pose_name:s1,s2,...,s8\|delay"` format. |
| 2.2 | `SLEEP:duration` steps pause without hardware commands. |
| 2.3 | If no explicit delay: auto-wait = `15.0 − (avg_speed − 1) × 2.4` seconds. |
| 2.4 | Sleeps execute in 0.1 s increments, checking stop/pause flags each tick. |
| 2.5 | In loop mode, a 0.5 s gap separates iterations. |
| 2.6 | Unknown pose names are skipped with a warning. |
| 2.7 | Play/Pause/Stop buttons toggle enabled/disabled state during execution. |

### FR-SEQ-3: Sequence Manager Dialog
Two-panel dialog accessed via "🔧 Manage" button.

| AC | Criterion |
|----|-----------|
| 3.1 | Left panel: listbox of saved sequences with Execute, Edit, Delete buttons. |
| 3.2 | Double-clicking executes the sequence once (non-looping) without closing the dialog. |
| 3.3 | Edit loads steps into the builder, pre-filling the name field. |

### FR-SEQ-4: Sequence Builder
Right panel for constructing sequences from poses.

| AC | Criterion |
|----|-----------|
| 4.1 | Available poses listed; double-click adds a step with current speed/delay. |
| 4.2 | Per-finger speed spinboxes (1–6); "⬇ Copy from UI" imports main window speeds. |
| 4.3 | Delay entry appends `\|delay` suffix to pose steps. |
| 4.4 | "⏱ Delay" inserts a standalone `SLEEP:Xs` step. |
| 4.5 | ↑/↓ reorder, ➖ remove, 🗑 clear all. |
| 4.6 | "💾 Save Sequence" validates name, saves, refreshes dropdowns. |
| 4.7 | "▶ Execute" runs the built sequence without saving or closing the dialog. |

### FR-SEQ-5: Delay Input Validation
Invalid float values in the delay entry are handled gracefully.

| AC | Criterion |
|----|-----------|
| 5.1 | Non-numeric delay defaults to no delay (step added without `\|delay`). |
| 5.2 | Non-numeric SLEEP delay shows "Invalid delay value" in the status bar. |

---

## 7. Servo Monitoring

### FR-MON-1: Background Telemetry Collection
A daemon thread polls all 8 servos at ~10 Hz.

| AC | Criterion |
|----|-----------|
| 1.1 | Thread sleeps 0.1 s between iterations. |
| 1.2 | Metrics collected per servo: position, load, temperature, voltage, speed, moving flag, status, goal. |
| 1.3 | A failed read repeats the last known value to keep arrays in sync. |
| 1.4 | Feedback data is updated under `feedback_lock` atomically. |

### FR-MON-2: Chart Display
Matplotlib chart embedded in the right panel.

| AC | Criterion |
|----|-----------|
| 2.1 | Selectable metrics: Position, Target vs Current, Torque, Speed, Temperature, Voltage, Moving. |
| 2.2 | "Servos" dropdown toggles which of 8 traces are visible (with ✓ All / ✕ None). |
| 2.3 | Chart redraws are throttled to ≥100 ms apart. |
| 2.4 | No metrics selected → "Select at least one metric" message. |
| 2.5 | No data → "Waiting for data..." message. |

### FR-MON-3: Chart Modes
Two modes: Multi-Servo and Scope.

| AC | Criterion |
|----|-----------|
| 3.1 | Scope mode shows a "Scope Servo" selector to focus on a single servo. |
| 3.2 | Multi-Servo hides the Scope Servo selector. |

### FR-MON-4: Chart Zoom & Pan
Four sliders for controlling the view.

| AC | Criterion |
|----|-----------|
| 4.1 | Y-Zoom: 0.2× to 5.0×, default 1.1×. |
| 4.2 | Y-Pan: −3.0 to +3.0, default 0.0. |
| 4.3 | Time-Zoom: 10 % to 100 % of available data. |
| 4.4 | Time-Pan: 0 % (earliest) to 100 % (latest). |
| 4.5 | All sliders trigger debounced chart redraws. |

### FR-MON-5: Rolling Mode
Limits chart to the latest N data points.

| AC | Criterion |
|----|-----------|
| 5.1 | When enabled and data exceeds `max_data_points` (100), oldest samples are discarded. |
| 5.2 | Disabling rolling retains all collected data. |

### FR-MON-6: Pause / Resume / Clear Chart

| AC | Criterion |
|----|-----------|
| 6.1 | Pause stops chart redraws; telemetry collection continues. |
| 6.2 | Clear resets all data arrays and zoom/pan to defaults. |

### FR-MON-7: Feedback Panel
Grid table showing live telemetry for all servos.

| AC | Criterion |
|----|-----------|
| 7.1 | Columns: S1–S8. Rows: Goal, Position, Speed, Torque, Voltage, Current, Temperature, Status, Moving. |
| 7.2 | Values formatted by `format_feedback_value()`: position `X.XX°`, speed `X.X°/s`, voltage `X.XX V`, temperature `X.X °C`, current `X mA`, load `X.X %`, status `0xHH`, moving `Yes/No`. |
| 7.3 | Only changed cells are updated (diff cache). |
| 7.4 | Refresh throttled to ≥50 ms between updates. |

---

## 8. Configuration

### FR-CFG-1: App Config Loading
`config.yaml` is loaded with defaults for all missing keys.

| AC | Criterion |
|----|-----------|
| 1.1 | Missing file → full default config used. |
| 1.2 | Missing keys merged from defaults (two-level merge). |
| 1.3 | Parse failure → defaults returned, error printed to stdout. |

### FR-CFG-2: Servo Mapping
Servo IDs per finger defined in `config.yaml` → `servos`.

| AC | Criterion |
|----|-----------|
| 2.1 | Config defines pointer=[1,2], middle=[3,4], ring=[5,6], thumb=[7,8]. |
| 2.2 | `all_ids` = [1,2,3,4,5,6,7,8]. |

### FR-CFG-3: Angle Limits

| AC | Criterion |
|----|-----------|
| 3.1 | `servo_min` = −40, `servo_max` = 110, `base_min` = 0, `base_max` = 110, `side_min` = −40, `side_max` = 40. |
| 3.2 | All slider ranges derive from these values. |

### FR-CFG-4: Auto Extremes
Bilinear interpolation endpoints for side-offset calculation.

| AC | Criterion |
|----|-----------|
| 4.1 | `left_open`, `right_open`, `left_closed`, `right_closed`, `center_open`, `center_closed` are configurable. |
| 4.2 | `compute_auto_positions()` uses these for interpolation. |

### FR-CFG-5: Speed Configuration

| AC | Criterion |
|----|-----------|
| 5.1 | `speeds.default` = 3, `speeds.min` = 1, `speeds.max` = 6. |

---

## 9. CLI Tool

### FR-CLI-1: List Poses and Sequences
`--list` prints all poses and sequences without opening a connection.

| AC | Criterion |
|----|-----------|
| 1.1 | Output shows pose count, each name with positions. |
| 1.2 | Output shows sequence count, each name with step count and details. |
| 1.3 | No serial connection is opened. |

### FR-CLI-2: Apply Pose
`--pose NAME` sends a saved pose to the hardware.

| AC | Criterion |
|----|-----------|
| 2.1 | Positions loaded from config; default speed 3 applied to all servos. |
| 2.2 | Unknown pose → error + `sys.exit(1)`. |

### FR-CLI-3: Play Sequence
`--sequence NAME` plays a sequence; `--loop` repeats until Ctrl+C.

| AC | Criterion |
|----|-----------|
| 3.1 | Speeds and delay parsed from step string. |
| 3.2 | `SLEEP` steps pause without hardware commands. |
| 3.3 | No explicit delay → auto-wait = `15.0 − (avg_speed − 1) × 2.4` seconds. |
| 3.4 | SIGINT sets `stop_flag` for graceful interrupt. |
| 3.5 | Unknown sequence → exit with error. |
| 3.6 | Empty sequence → exit with error. |
| 3.7 | Unknown poses within a sequence are skipped with a WARNING. |

### FR-CLI-4: Step Parsing
`parse_step()` handles multiple formats.

| AC | Criterion |
|----|-----------|
| 4.1 | `SLEEP:2.0s` → sleep of 2.0 s. |
| 4.2 | `open:3,3,...\|2.0s` → pose "open" with speeds and 2.0 s delay. |
| 4.3 | `open` (bare name) → pose with default speeds, no delay. |
| 4.4 | Speeds shorter than 8 padded with 3; longer truncated. |
| 4.5 | Duration suffix `s` / `S` stripped. |

### FR-CLI-5: Mutually Exclusive Actions
`--list`, `--pose`, and `--sequence` are mutually exclusive.

| AC | Criterion |
|----|-----------|
| 5.1 | Passing multiple actions → non-zero exit. |
| 5.2 | `--loop` without `--sequence` → error. |

### FR-CLI-6: Config File Override
`--config PATH` uses an alternative YAML file.

| AC | Criterion |
|----|-----------|
| 6.1 | Missing file → error + `sys.exit(1)`. |

---

## 10. Data Persistence

### FR-DATA-1: YAML Config File
Poses and sequences are stored in `data/hand_config.yaml`.

| AC | Criterion |
|----|-----------|
| 1.1 | File uses YAML format with `poses` and `sequences` top-level keys. |

### FR-DATA-2: Load Config

| AC | Criterion |
|----|-----------|
| 2.1 | Missing file → `{'poses': {}, 'sequences': {}}`. |
| 2.2 | Empty file → keys auto-populated. |
| 2.3 | Malformed YAML → empty structure, error printed to stdout. |

### FR-DATA-3: Save Config with Inline Arrays
Positions are saved in flow style via regex post-processing.

| AC | Criterion |
|----|-----------|
| 3.1 | File contains `positions: [v1, v2, …, v8]` style. |
| 3.2 | Negative values preserved in inline format. |
| 3.3 | Returns `True` on success, `False` on error. |

### FR-DATA-4: Data Directory Auto-Creation

| AC | Criterion |
|----|-----------|
| 4.1 | `data/` directory is created if it does not exist before writing. |

### FR-DATA-5: Round-Trip Integrity
Data written by the GUI can be read by the CLI and vice versa.

| AC | Criterion |
|----|-----------|
| 5.1 | Poses, negative positions, and sequence steps survive a GUI-save → CLI-read round trip. |

---

## 11. Error Handling

### FR-ERR-1: Connection Error
Failed connections do not crash the application.

| AC | Criterion |
|----|-----------|
| 1.1 | Status bar shows "Connection failed: …"; `connected` stays `False`. |

### FR-ERR-2: Invalid Baudrate

| AC | Criterion |
|----|-----------|
| 2.1 | Non-numeric baudrate → status bar shows "Invalid baudrate". |

### FR-ERR-3: Monitor Thread Recovery

| AC | Criterion |
|----|-----------|
| 3.1 | A single servo read failure does not crash the thread. |
| 3.2 | Errors are printed to stdout. |

### FR-ERR-4: Moving Flag Degradation

| AC | Criterion |
|----|-----------|
| 4.1 | After 3 consecutive `read_moving` failures, supervision is disabled with a log message. |
| 4.2 | Falls back from `sync_read_moving` to per-servo reads on first sync failure. |

### FR-ERR-5: Pose Completion Warnings

| AC | Criterion |
|----|-----------|
| 5.1 | Target vs. actual error > 5° triggers a ⚠ warning in the log. |
| 5.2 | Movement timeout (6.0 s) triggers a timeout warning if servos never stop moving. |

### FR-ERR-6: Missing Config (CLI)

| AC | Criterion |
|----|-----------|
| 6.1 | Missing config file → error message + `sys.exit(1)`. |

### FR-ERR-7: Empty / Invalid Sequence

| AC | Criterion |
|----|-----------|
| 7.1 | Empty sequence steps → `sys.exit(1)`. |
| 7.2 | Unknown poses in sequence → skipped with WARNING. |

---

## 12. UI Layout

### FR-UI-1: Window Structure

| AC | Criterion |
|----|-----------|
| 1.1 | Title includes version: "AmazingHand Controller v0.8". |
| 1.2 | Initial geometry: 1920×1200. |
| 1.3 | Horizontal `PanedWindow` separates left (controls) and right (chart) panels. |

### FR-UI-2: Left Panel

| AC | Criterion |
|----|-----------|
| 2.1 | Row 1: Ring, Middle, Pointer (3 fingers across). |
| 2.2 | Row 2: Thumb (right) + stacked controls (Connection, Global, Pose, Sequence). |
| 2.3 | Execution log below controls in a resizable vertical splitter. |

### FR-UI-3: Status Bar

| AC | Criterion |
|----|-----------|
| 3.1 | Updates on connect, disconnect, finger selection, speed change, pose operations, and errors. |

### FR-UI-4: Execution Log

| AC | Criterion |
|----|-----------|
| 4.1 | Messages prefixed with `[HH:MM:SS.mmm]` timestamp. |
| 4.2 | Auto-scrolls to the latest entry. |
| 4.3 | Messages also printed to stdout. |

### FR-UI-5: Tooltips

| AC | Criterion |
|----|-----------|
| 5.1 | Yellow popup appears after 500 ms hover, positioned below-right of widget. |
| 5.2 | Disappears on mouse leave or button press. |

### FR-UI-6: Right Panel (Chart Area)

| AC | Criterion |
|----|-----------|
| 6.1 | Vertical `PanedWindow`: chart on top (min 200 px), feedback below (min 150 px). |
| 6.2 | Time sliders below chart; Y sliders to the right. |

### FR-UI-7: CLI Help

| AC | Criterion |
|----|-----------|
| 7.1 | `--help` exits 0 and shows all options. |

### FR-UI-8: GUI Command-Line Arguments

| AC | Criterion |
|----|-----------|
| 8.1 | `--port` overrides default serial port. |
| 8.2 | `--baudrate` overrides default baudrate (1000000). |

---

## Test Coverage Summary

| Test File | Scope | Count |
|-----------|-------|-------|
| `tests/test_hand_logic.py` | `load_app_config` (6), `servo_mapping` (3), `angle_limits` (1), `auto_extremes` (2), `speed_config` (1), `default_serial_port` (2), `ensure_data_dir` (2), `load_pose_definitions` (3), `compute_auto_positions` (6), `decompose_servo_positions` (5), `coerce_numeric` (11), `coerce_angle_degrees` (6), `coerce_bool` (6), `load_to_percent` (7), `estimate_current_from_load` (5), `format_feedback_value` (12), `get_time_window_indices` (8), `angle_rad` (3) | 89 |
| `tests/test_gui_utils.py` | `validate_name` (21), `clamp` (8), `load_config` (6), `save_config` (6) | 41 + 10 parametrized |
| `tests/test_cmd.py` | `angle_rad` (8), `parse_step` (11), `cmd_list` (4), `load_config` (2), `apply_pose` (6), `_interruptible_sleep` (2), `auto_wait` (3), `mutual_exclusion` (1), `config_override` (2) | 41 |
| `tests/test_integration.py` | `cmd_pose` e2e (5), `cmd_sequence` e2e (6), config round-trip (3) | 14 |
| `tests/test_system.py` | CLI subprocess: `--help` (5), `--list` (9), `--help` options (3), error paths (5) | 22 |
| `tests/test_system_hardware.py` | Real hardware: connect (2), CLI connection (2), pose apply (3), speed (2), telemetry (6), sequence (2), error recovery (1), movement (2), disconnect (1) — **requires `--hardware` flag** | 21 |
| **Total (no hardware)** | **217 tests** |
| **Total (with hardware)** | **238 tests** |

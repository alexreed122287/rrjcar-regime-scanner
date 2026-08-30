# Copilot instructions — rrjcar-regime-scanner

This repository is an HMM regime-detection engine wired to a **live Tradier brokerage account**.
Code here can place real orders. Treat every change as safety-critical.

## Hard rules — never violate

1. **Never delete, move, rename, truncate, or auto-clear the `HALT_TRADING` sentinel file**, and never
   add a code path that clears it. It exists so a human must manually inspect a drawdown event and
   consciously re-enable trading. If asked to "reset risk state" or "make the tests pass" and the
   obvious fix is removing the sentinel, stop and say so instead.
2. **Never widen, raise, disable, or bypass a circuit-breaker threshold** in `risk_manager.py`
   (daily -2% / -3%, weekly -5%, peak drawdown -10%) unless the request explicitly and specifically
   asks for that threshold to change. Do not "tune" them as a side effect of another task.
3. **Never remove or weaken the risk check in `api/routes_broker.py`'s order paths.** New order
   endpoints must call the risk gate before submitting.
4. **Exits are always allowed; entries are gated.** Any risk logic must permit closing/reducing
   existing risk even when new entries are blocked. Never invert this.
5. **Never hardcode credentials.** Tradier tokens come from env vars or `.tradier_settings.json`
   (gitignored). Never write a token, account id, or key into tracked source, tests, or fixtures.

## Order-capable code — manual review required

These files can move real money. Do not edit them as an incidental part of an unrelated task, and
never auto-run them:

- `order_executor.py`
- `tradier_broker.py` (`place_equity_order`, `place_option_order`, `cancel_order`)
- `api/routes_broker.py`
- `roll_manager.py`

Propose diffs for these and let a human review before they land. Do not add them to any
auto-approve, task-runner, or watch configuration.

## Look-ahead bias

Regime labels used for **live decisions or out-of-sample evaluation must be causal**. Bar `t`'s
regime may only depend on bars `<= t`.

- `RegimeDetector.train()` uses full-sequence Viterbi — that is acceptable for historical labeling
  and in-sample fitting only.
- `RegimeDetector.predict_current()` and `filtered_regimes()` use the forward algorithm and must
  stay causal. Do not "simplify" them back to `model.predict()` or `predict_proba()` —
  `predict_proba` is forward-backward smoothed and leaks future information.
- `walk_forward.py` must never fit a model on data that overlaps its own out-of-sample window.

Any change touching these paths needs a test proving that appending future bars does not alter
previously assigned labels.

## Conventions

- Additive over invasive: prefer new modules and optional keyword arguments with
  behavior-preserving defaults over rewriting working code paths.
- Every new module gets tests under `tests/`. Run `pytest` before declaring done.
- Sentinel/state files (`HALT_TRADING`, `.equity_history.json`) belong in `.gitignore`, never
  committed with real account data.

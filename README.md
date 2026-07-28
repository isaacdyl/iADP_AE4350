# iADP_AE4350

# iADP Longitudinal Flight Controller (AE4350)

Incremental Approximate Dynamic Programming (iADP) controller for longitudinal
pitch-rate tracking on a nonlinear short-period aircraft model, implemented for
the AE4350 Bio-Inspired Intelligence assignment. Method mostly follows
Konatala et al. 2024, with some documented deviations (see the report).

Online RLS incremental-model identification, model-based value iteration for
policy evaluation, closed-form policy improvement, and an open-loop
re-identification manoeuvre for fault recovery.

## Layout

- `+iadp/` — core package
  - `ShortPeriodAircraft.m` — truth plant (q, alpha), nonlinear pitching moment,
    first-order actuator, fault hooks (effectiveness loss, sign reversal, c.g. shift)
  - `IncrementalRLS.m` — recursive least-squares identification of the incremental model
  - `KernelPolicyEvaluator.m` — value-function kernel: value iteration (default)
    or data-driven Kronecker-LS (kept for comparison, see report §3.2)
  - `policyImprovement.m` — closed-form incremental control law
  - `solveDiscountedRiccati.m`, `optimalGain.m` — LQR benchmark on the true/identified model
  - `referenceSignal.m` — repeating 3-2-1-1 multistep command
  - `runSimulation.m` — main closed-loop simulation (name/value options, see header)
- `experiments/`
  - `step1_test_openloop_id.m`, `step2a_test_rls_synthetic.m`,
    `step2b_test_rls_warmstart_manual.m`, `step3_test_riccati.m` — unit-test-style
    scripts used during development to validate identification and the Riccati solve
  - `exp1_nominal_tracking.m` — nominal tracking + online learning curves
  - `exp2_lqr_convergence.m` — convergence of the learned kernel/gain to the LQR optimum
  - `exp3_fault_injection.m` — fault robustness: three fault types, recovery boundary
    sweep, and with/without re-identification comparison
  - `exp4_sensitivity_sweep.m` — Monte-Carlo sensitivity to discount factor, cost
    ratio, excitation duration, and excitation amplitude (8 seeds per condition)
- `results/` — `.mat` outputs from each experiment (seed 1, or seeds 1–8 for Exp. 4)
- `figures/` — figures as used in the report

## Reproduce

```matlab
addpath(genpath(pwd));
run('experiments/exp1_nominal_tracking.m')
run('experiments/exp2_lqr_convergence.m')
run('experiments/exp3_fault_injection.m')
run('experiments/exp4_sensitivity_sweep.m')
```

All experiments are deterministic at `seed = 1` (Exp. 4 additionally sweeps seeds 1–8).

## Key results

- Incremental model identified online to 4.2% (`G̃`) / 3.1% (`F̃`) relative error from sensor data alone.
- Learned value-function kernel converges to within 5.1% of the true discounted-LQR optimum.
- Learned feedback gain `[-7.08, 0.51, 7.01]` vs. true optimal `[-6.93, 0.25, 6.99]` — matches on the
  cost-relevant channels (q, q_ref) to 2%; the larger relative error on the uncosted α channel has
  negligible effect on closed-loop tracking
- Recovers online from 60% elevator-effectiveness loss, c.g. shifts, and full elevator sign
  reversal, via a brief open-loop re-identification manoeuvre and model re-fit — post-fault RMSE
  drops from 23.0 to 3.1 deg/s in the sign-reversal case.
- Nominal settled tracking RMSE ~1.1 deg/s; sensitivity to discount factor, cost ratio, and
  excitation parameters characterised over 8 seeds per condition.

See the accompanying report for the full method, the rationale behind each design decision
(SLA model-freezing, model-based value iteration in place of the paper's data-driven kernel
fit, and the open-loop fault-recovery manoeuvre), and the discussion of where the method's
assumptions break down.

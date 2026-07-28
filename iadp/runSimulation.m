function hist = runSimulation(varargin)
% RUNSIMULATION  Closed-loop iADP pitch-rate controller (longitudinal).
%
% Multi-rate: plant integrated at 1 kHz (RK4 + actuator), controller / RLS /
% critic at 100 Hz with ZOH on the elevator cmd.
%
% Phases:
%   1) excitation (0..learn_duration): open-loop multisine only, cold-start
%      RLS identifies F_tilde, G_tilde. ID at 100 Hz
%   2) tracking: model frozen (SLA), actor runs the incremental law, critic
%      does 20 Hz value-iteration backups from P0 = I.
%   3) after a fault: cov reset + open-loop re-ID manoeuvre, then freeze again.

    p = inputParser;
    addParameter(p, 'T_total', 24.0);
    addParameter(p, 'dt', 1e-3);                    % plant step
    addParameter(p, 'ctrl_rate_hz', 100.0);         % controller / ID rate
    addParameter(p, 'critic_rate_hz', 20.0);        % Bellman backup rate
    addParameter(p, 'gamma_rls', 0.999);
    addParameter(p, 'gamma', 0.9);
    addParameter(p, 'Q', 100);
    addParameter(p, 'R', 1);
    addParameter(p, 'cov0', 1e8);                   % large -> cold start acts like batch LS
    addParameter(p, 'fault_time', []);
    addParameter(p, 'fault_factor', 1.0);
    addParameter(p, 'fault_type', 'effectiveness'); % 'effectiveness' | 'malpha'
    addParameter(p, 'seed', 0);
    addParameter(p, 'ref_amp_deg', 5.0);
    addParameter(p, 'excite_amp_deg', 1.0);
    addParameter(p, 'learn_duration', 3.0);         % excitation-only ID phase
    addParameter(p, 'meas_noise_q_deg', 0.0002);
    addParameter(p, 'meas_noise_alpha_deg', 0.0005);
    addParameter(p, 'filter_cutoff_hz', 25.0);
    addParameter(p, 'init_P_from_riccati', false);  % ablation only
    addParameter(p, 'pe_method', 'value_iteration'); % 'value_iteration' | 'kron_ls'
    addParameter(p, 'model_mode', 'sla');            % 'sla' | 'continuous'
    addParameter(p, 'relearn_after_fault', 3.0);
    addParameter(p, 'relearn_excite_amp_deg', 4.0);
    addParameter(p, 'relearn_damp_gain', 0.0);
    parse(p, varargin{:});
    o = p.Results;

    rng(o.seed);
    dt   = o.dt;
    dec  = round((1/o.ctrl_rate_hz)/dt);   % sim steps per controller tick
    dt_c = dec*dt;                         % controller period (10 ms)
    n_ticks = floor(o.T_total/dt_c);
    n_steps = n_ticks*dec;
    t = (0:n_steps-1)'*dt;
    critic_decim = max(1, round(o.ctrl_rate_hz/o.critic_rate_hz));

    ref_t0 = o.learn_duration + 0.2;
    q_ref = iadp.referenceSignal(t, 'amp', deg2rad(o.ref_amp_deg), ...
                'unit', 0.7, 'period', 8.0, 't0', ref_t0, ...
                'n_repeats', ceil((o.T_total-ref_t0)/8) + 1);

    plant = iadp.ShortPeriodAircraft(dt);
    plant.reset();

    n_X = 3;
    C_E = [1, 0, -1];
    Q_aug = o.Q * (C_E' * C_E);
    R_mat = o.R;

    % multisine excitation freqs (rad/s)
    freqs = 2*pi*[0.7, 1.3, 2.1, 3.4];

    % sensor noise + 1st-order low-pass on the 1 kHz measurements
    nq_std = deg2rad(o.meas_noise_q_deg);
    na_std = deg2rad(o.meas_noise_alpha_deg);
    tau_f = 1/(2*pi*o.filter_cutoff_hz);
    a_lp = dt/(tau_f+dt);
    q_f = 0; a_f = 0;

    rls = iadp.IncrementalRLS(2, 1, o.gamma_rls, o.cov0);
    kpe = iadp.KernelPolicyEvaluator(n_X, o.gamma, 40, 1e-6, eye(n_X));

    % logging
    hist.t = t; hist.qref = q_ref;
    hist.q = zeros(n_steps,1); hist.alpha = zeros(n_steps,1);
    hist.delta = zeros(n_steps,1); hist.delta_cmd = zeros(n_steps,1);
    hist.cost = zeros(n_steps,1); hist.fault_factor = ones(n_steps,1);
    hist.t_tick = (0:n_ticks-1)'*dt_c;
    hist.Ferr_rel = nan(n_ticks,1); hist.Gerr_rel = nan(n_ticks,1);
    hist.Pnorm_tick = zeros(n_ticks,1);
    hist.F_hist = zeros(n_ticks,2,2); hist.G_hist = zeros(n_ticks,2);
    max_backups = ceil(n_ticks/critic_decim) + 1;
    hist.P_log = zeros(n_X, n_X, max_backups);
    hist.t_backup = zeros(max_backups,1);
    n_bk = 0;

    % tick-level state
    X_prev   = [0; 0; q_ref(1)];   % augmented state at prev tick
    dX_prev  = zeros(3,1);         % prev increment (RLS regressor)
    delta_prev  = 0;               % achieved elevator at prev tick
    ddelta_prev = 0;
    fault_applied = false;
    P_initialised = false;

    for i = 1:n_ticks
        k0 = (i-1)*dec + 1;        % sim index at start of tick
        t_now = t(k0);

        % fault injection (tick granularity)
        if ~isempty(o.fault_time) && t_now >= o.fault_time && ~fault_applied
            switch o.fault_type
                case 'effectiveness', plant.injectFault(o.fault_factor);
                case 'malpha',        plant.setMalpha(plant.Malpha * o.fault_factor);
            end
            fault_applied = true;
            rls.Phi = eye(rls.n_w) * o.cov0;   % forget old confidence
        end

        % phase logic
        in_learn  = t_now < o.learn_duration;
        in_relearn = fault_applied && t_now < (o.fault_time + o.relearn_after_fault);
        if strcmp(o.model_mode, 'continuous')
            id_active = true;
        else
            id_active = in_learn || in_relearn;   % SLA: only these windows
        end

        % measure (filtered) and form increments
        X_t  = [q_f; a_f; q_ref(k0)];
        dX_t = X_t - X_prev;

        % RLS update with the previous tick's transition
        if id_active && i > 2
            W = [dX_prev(1:2); ddelta_prev];
            rls.update(dX_t(1:2), rls.predict(W), W);
        end

        [F_t, G_t] = rls.FG();
        F_aug = [F_t, zeros(2,1); 0, 0, 1];
        G_aug = [G_t; 0];

        % critic init at end of excitation phase
        if ~in_learn && ~P_initialised
            if o.init_P_from_riccati
                kpe.reset(iadp.solveDiscountedRiccati(F_aug, G_aug, Q_aug, R_mat, o.gamma));
            else
                kpe.reset(eye(n_X));
            end
            hist.P0 = kpe.P;
            P_initialised = true;
        end

        % critic: one VI backup at the critic rate
        do_backup = P_initialised && mod(i-1, critic_decim) == 0;
        if strcmp(o.model_mode, 'sla') && in_relearn
            do_backup = false;                 % model in flux, hold P
        end
        if do_backup && strcmp(o.pe_method, 'value_iteration')
            kpe.bellmanBackup(F_aug, G_aug, Q_aug, R_mat);
            n_bk = n_bk + 1;
            hist.P_log(:,:,n_bk) = kpe.P;
            hist.t_backup(n_bk) = t_now;
        end

        % actor: elevator cmd for this tick
        excite = 0;
        for jj = 1:numel(freqs)
            excite = excite + sin(freqs(jj)*t_now + (jj-1));
        end
        excite = excite/numel(freqs);

        if in_learn
            delta_cmd = deg2rad(o.excite_amp_deg) * excite;          % open loop
        elseif in_relearn && strcmp(o.model_mode, 'sla')
            delta_cmd = deg2rad(o.relearn_excite_amp_deg) * excite ...
                        - o.relearn_damp_gain * plant.q;             % open-loop re-ID
        else
            dDelta = iadp.policyImprovement(F_aug, G_aug, kpe.P, X_t, dX_t, ...
                                            delta_prev, R_mat, o.gamma);
            delta_cmd = delta_prev + dDelta(1);
        end

        % kron_ls option: data-driven PE, fragile on this cost, kept for comparison
        if strcmp(o.pe_method, 'kron_ls') && P_initialised && i > 2
            e_prev = C_E * X_prev;
            c_prev = o.Q*e_prev^2 + R_mat*delta_prev^2;
            kpe.addSample(X_prev, X_t, c_prev);
        end

        % hold the cmd over the sim substeps
        for j = 1:dec
            k = k0 + j - 1;
            [xnext, delta_true] = plant.step(delta_cmd);
            q_m = xnext(1) + nq_std*randn();
            a_m = xnext(2) + na_std*randn();
            q_f = q_f + a_lp*(q_m - q_f);
            a_f = a_f + a_lp*(a_m - a_f);

            e_k = plant.q - q_ref(k);
            hist.q(k) = plant.q; hist.alpha(k) = plant.alpha;
            hist.delta(k) = delta_true; hist.delta_cmd(k) = delta_cmd;
            hist.cost(k) = o.Q*e_k^2 + R_mat*delta_true^2;
            hist.fault_factor(k) = plant.fault_factor;
        end

        % tick-rate logging: model error vs true 10 ms linearisation
        [A,B] = plant.linearization();
        F_true = eye(2) + A*dt_c; G_true = B*dt_c;
        hist.Ferr_rel(i) = norm(F_t - F_true,'fro')/norm(F_true,'fro');
        hist.Gerr_rel(i) = norm(G_t - G_true)/norm(G_true);
        hist.Pnorm_tick(i) = norm(kpe.P,'fro');
        hist.F_hist(i,:,:) = F_t; hist.G_hist(i,:) = G_t';

        % roll tick state forward
        ddelta_prev = delta_true - delta_prev;   % achieved elevator increment
        delta_prev  = delta_true;
        dX_prev     = dX_t;
        X_prev      = X_t;
    end

    % outputs
    hist.P_log = hist.P_log(:,:,1:n_bk);
    hist.t_backup = hist.t_backup(1:n_bk);
    hist.n_backups = n_bk;
    hist.P_final = kpe.P; hist.Q_aug = Q_aug;
    [Ff,Gf] = rls.FG();
    Ff_aug = [Ff, zeros(2,1); 0,0,1]; Gf_aug = [Gf;0];
    hist.F_final = Ff; hist.G_final = Gf;
    hist.P_star_final = iadp.solveDiscountedRiccati(Ff_aug, Gf_aug, Q_aug, R_mat, o.gamma);
    hist.n_kernel_projected = kpe.n_projected;
    hist.n_kernel_solved = kpe.n_solved;
    hist.n_kernel_skipped = kpe.n_skipped;
    hist.plant = plant;
    hist.dt_c = dt_c;
    hist.n_learn_ticks = round(o.learn_duration/dt_c);
    hist.options = o;
end

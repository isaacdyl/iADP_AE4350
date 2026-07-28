%% Step 2g: warm-start RLS from a short batch OLS fit, then continue recursively
clear; clc;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
dt = 1e-3; T_warm = 2.0; T_total = 10.0;
n_warm = round(T_warm/dt); n_total = round(T_total/dt);
t = (0:n_total-1)'*dt;
freqs = 2*pi*[0.7, 1.3, 2.1, 3.4];

plant = iadp.ShortPeriodAircraft(dt); plant.reset();

% --- Phase 1: pure excitation, collect data for batch warm-start ---
X_hist = zeros(n_warm,2); delta_hist = zeros(n_warm,1);
for k = 1:n_warm
    s = 0; for i = 1:numel(freqs), s = s + sin(freqs(i)*t(k) + (i-1)); end
    delta_cmd = deg2rad(2.0) * (s/numel(freqs));
    [xnext, delta_true] = plant.step(delta_cmd);
    X_hist(k,:) = xnext'; delta_hist(k) = delta_true;
end
dXw = diff(X_hist); dXw_t = dXw(1:end-1,:); dXw_next = dXw(2:end,:);
ddw = diff(delta_hist); ddw = ddw(1:end-1);
W_batch = [dXw_t, ddw];
Theta_init = W_batch \ dXw_next;
resid = dXw_next - W_batch*Theta_init;
fprintf('Batch warm-start residual std: %.3e\n', std(resid(:)));

% --- Phase 2: continue with recursive RLS, warm-started, for the rest of the run ---
rls = iadp.IncrementalRLS(2, 1, 0.999, 10);   % modest cov0 now, since Theta already good
rls.Theta = Theta_init;

X_prev = X_hist(end-1,:)'; delta_prev = delta_hist(end);
Gnorm_err = zeros(n_total - n_warm, 1);
for k = (n_warm+1):n_total
    X_t = [plant.q; plant.alpha]; dX_t = X_t - X_prev;
    s = 0; for i = 1:numel(freqs), s = s + sin(freqs(i)*t(k) + (i-1)); end
    delta_cmd = deg2rad(2.0) * (s/numel(freqs));
    [xnext, delta_true] = plant.step(delta_cmd);
    ddelta_actual = delta_true - delta_prev;
    X_next = [xnext(1); xnext(2)]; dX_next_meas = X_next - X_t;
    W = [dX_t; ddelta_actual];
    dX_pred = rls.predict(W);
    rls.update(dX_next_meas, dX_pred, W);
    [F_t, G_t] = rls.FG(); [A,B] = plant.linearization(); G_true = B*dt;
    Gnorm_err(k-n_warm) = norm(G_t - G_true)/norm(G_true);
    X_prev = X_t; delta_prev = delta_true;
end
[F_final, G_final] = rls.FG();
fprintf('relerr right after warm-start: %.4f\n', Gnorm_err(1));
fprintf('relerr at end (t=10s): %.4f\n', Gnorm_err(end));
disp('G_final ='); disp(G_final); disp('G_true ='); disp(B*dt);
disp('F_final ='); disp(F_final);

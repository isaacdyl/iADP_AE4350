%% Experiment 3: Robustness - online re-adaptation to faults + recovery boundary
clear; clc; close all;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
figdir = 'D:\Bio_inspired\iADP_Control_Assignment\figures\';
resdir = 'D:\Bio_inspired\iADP_Control_Assignment\results\';
dt = 1e-3;
fault_t = 12.0;

% --- 3a: three fault types, time histories ---
cases = {
  struct('type','effectiveness','factor',0.4,'name','60% elevator effectiveness loss');
  struct('type','malpha','factor',1.6,'name','c.g. shift (Malpha x1.6, more stable)');
  struct('type','malpha','factor',0.4,'name','c.g. shift (Malpha x0.4, less stable)');
};
f = figure('Position',[100 100 900 700],'Color','w');
for c = 1:numel(cases)
    cc = cases{c};
    h = iadp.runSimulation('T_total', 22.0, ...
        'fault_time', fault_t, 'fault_type', cc.type, 'fault_factor', cc.factor, ...
        'relearn_after_fault', 3.0, 'seed', 1);
    t = h.t;
    subplot(numel(cases),1,c);
    plot(t, rad2deg(h.qref),'k--','LineWidth',1.0); hold on;
    plot(t, rad2deg(h.q),'b','LineWidth',0.9);
    xline(fault_t,'r-','fault','LineWidth',1.2);
    xline(fault_t+3.0,'m:','relearn end');
    ylabel('q [deg/s]'); title(cc.name); grid on;
    if c==1, legend('q_{ref}','q','Location','best'); end
    if c==numel(cases), xlabel('time [s]'); end
    rmse_post = rad2deg(sqrt(mean((h.q(round((fault_t+4)/dt):end)-h.qref(round((fault_t+4)/dt):end)).^2)));
    fprintf('%-45s post-recovery RMSE = %.2f deg/s, dmax=%.1f deg\n', cc.name, rmse_post, rad2deg(max(abs(h.delta))));
end
exportgraphics(f, [figdir 'exp3_fault_types.png'], 'Resolution', 150);

% --- 3b: recovery-boundary sweep over effectiveness-loss severity ---
factors = [1.0 0.8 0.6 0.4 0.3 0.2 0.1 0.05 -0.1 -0.3 -0.5];
rmse_rec = zeros(size(factors)); recovered = false(size(factors));
for i = 1:numel(factors)
    h = iadp.runSimulation('T_total', 22.0, ...
        'fault_time', fault_t, 'fault_type','effectiveness','fault_factor', factors(i), ...
        'relearn_after_fault', 3.0, 'seed', 1);
    rmse_rec(i) = rad2deg(sqrt(mean((h.q(round((fault_t+4)/dt):end)-h.qref(round((fault_t+4)/dt):end)).^2)));
    recovered(i) = rad2deg(max(abs(h.delta))) < 24.9 && rmse_rec(i) < 8;
end
f2 = figure('Position',[100 100 800 350],'Color','w');
plot(factors, rmse_rec,'o-','LineWidth',1.3,'MarkerFaceColor','b'); hold on;
yline(8,'r--','recovery threshold'); xline(0,'k:','sign reversal');
set(gca,'XDir','reverse');
xlabel('elevator effectiveness factor (1=healthy, <0 = sign reversal)');
ylabel('post-recovery RMSE [deg/s]'); grid on;
title('Recovery boundary: tracking error vs fault severity');
exportgraphics(f2, [figdir 'exp3_recovery_boundary.png'], 'Resolution', 150);

% --- 3c: learning-enabled recovery (with vs without online re-ID) ---
factors2 = [0.6 0.4 0.2 -0.2 -0.5];
rmse_with = zeros(size(factors2)); rmse_without = zeros(size(factors2));
for i = 1:numel(factors2)
    hw = iadp.runSimulation('T_total',24,...
        'fault_time',fault_t,'fault_type','effectiveness','fault_factor',factors2(i), ...
        'relearn_after_fault',3.0,'seed',1);
    hn = iadp.runSimulation('T_total',24,...
        'fault_time',fault_t,'fault_type','effectiveness','fault_factor',factors2(i), ...
        'relearn_after_fault',0.0,'seed',1);
    rr = @(h) rad2deg(sqrt(mean((h.q(round((fault_t+4)/dt):end)-h.qref(round((fault_t+4)/dt):end)).^2)));
    rmse_with(i) = rr(hw); rmse_without(i) = min(rr(hn), 99);
end
f3 = figure('Position',[100 100 800 380],'Color','w');
b = bar(categorical(string(factors2)), [rmse_with(:) rmse_without(:)]); grid on;
b(1).FaceColor=[0.2 0.5 0.9]; b(2).FaceColor=[0.85 0.3 0.3];
legend('with online re-ID','without re-ID (frozen model)','Location','northwest');
xlabel('elevator effectiveness factor'); ylabel('post-fault RMSE [deg/s]');
title('Online re-identification is what enables fault recovery');
exportgraphics(f3, [figdir 'exp3_learning_enabled_recovery.png'], 'Resolution', 150);
for i=1:numel(factors2)
  fprintf('factor %+.1f : with re-ID %.2f | without %.2f deg/s\n', factors2(i), rmse_with(i), rmse_without(i));
end

save([resdir 'exp3_faults.mat'], 'factors','rmse_rec','recovered');
fprintf('Exp3 done. Recovered for factors: %s\n', mat2str(factors(recovered)));

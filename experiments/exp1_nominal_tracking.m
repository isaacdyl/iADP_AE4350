%% Experiment 1: Nominal pitch-rate tracking + online learning curves
clear; clc; close all;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
figdir = 'D:\Bio_inspired\iADP_Control_Assignment\figures\';
resdir = 'D:\Bio_inspired\iADP_Control_Assignment\results\';

h = iadp.runSimulation('T_total', 24.0, 'seed', 1);
save([resdir 'exp1_nominal.mat'], 'h');

t = h.t; ld = h.options.learn_duration;
[A,B] = h.plant.linearization();
F_true = [eye(2)+A*h.dt_c, zeros(2,1); 0 0 1]; G_true = [B*h.dt_c; 0];
P_star = iadp.solveDiscountedRiccati(F_true, G_true, h.Q_aug, h.options.R, h.options.gamma);

% --- Figure 1: tracking + control ---
f1 = figure('Position',[100 100 800 600],'Color','w');
subplot(3,1,1);
plot(t, rad2deg(h.qref),'k--','LineWidth',1.2); hold on;
plot(t, rad2deg(h.q),'b','LineWidth',1.0);
xline(ld,'r:','excitation end','LabelOrientation','horizontal');
ylabel('pitch rate q [deg/s]'); legend('q_{ref}','q','Location','best');
title('iADP nominal pitch-rate tracking'); grid on;
subplot(3,1,2);
plot(t, rad2deg(h.q - h.qref),'m','LineWidth',0.8);
ylabel('tracking error [deg/s]'); grid on;
subplot(3,1,3);
plot(t, rad2deg(h.delta),'Color',[0 0.5 0],'LineWidth',1.0);
ylabel('elevator \delta_e [deg]'); xlabel('time [s]'); grid on;
exportgraphics(f1, [figdir 'exp1_tracking.png'], 'Resolution', 150);

% --- Figure 2: online learning curves ---
f2 = figure('Position',[100 100 800 500],'Color','w');
subplot(2,1,1);
plot(h.t_tick, h.Gerr_rel,'b','LineWidth',1.0); hold on;
plot(h.t_tick, h.Ferr_rel,'r','LineWidth',1.0);
xline(ld,'k:','model frozen (SLA)','LabelOrientation','horizontal');
ylabel('rel. model error'); legend('||G-G_{true}||/||G_{true}||','||F-F_{true}||/||F_{true}||');
title('Recursive-RLS model identification (cold start, 100 Hz)'); grid on; ylim([0 1]);
subplot(2,1,2);
plot(h.t_tick, h.Pnorm_tick,'k','LineWidth',1.0); hold on;
yline(norm(P_star,'fro'),'g--','||P*|| (true-model optimum)');
ylabel('||P||_F'); xlabel('time [s]');
title('Kernel norm: online value iteration from P_0 = I'); grid on;
exportgraphics(f2, [figdir 'exp1_learning.png'], 'Resolution', 150);

% --- metrics ---
e = rad2deg(h.q - h.qref);
trk = t >= ld + 0.2;
ch = [true; diff(h.qref)~=0]; settle = true(size(t));
for k = find(ch)', settle(t>=t(k) & t<t(k)+0.5) = false; end
settle = settle & trk;
fprintf('Exp1: full RMSE %.2f | settled %.2f deg/s\n', rms(e(trk)), rms(e(settle)));
fprintf('      early (%.1f-8s) %.2f | late (16-24s) %.2f deg/s\n', ld+0.2, ...
    rms(e(t>=ld+0.2 & t<8)), rms(e(t>=16)));
fprintf('      max |delta| %.1f deg | final Ferr %.2f%% Gerr %.2f%%\n', ...
    max(abs(rad2deg(h.delta))), 100*h.Ferr_rel(end), 100*h.Gerr_rel(end));
fprintf('      ||P_final-P*||/||P*|| = %.4f (%d backups)\n', ...
    norm(h.P_final-P_star,'fro')/norm(P_star,'fro'), h.n_backups);

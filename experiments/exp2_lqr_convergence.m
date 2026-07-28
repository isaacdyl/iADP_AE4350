%% Experiment 2: What did the agent learn? Online convergence to the LQR optimum.
clear; clc; close all;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
figdir = 'D:\Bio_inspired\iADP_Control_Assignment\figures\';
resdir = 'D:\Bio_inspired\iADP_Control_Assignment\results\';

h = iadp.runSimulation('T_total', 24.0, 'seed', 1);

% --- true-model optimal kernel & gain (benchmark) ---
[A,B] = h.plant.linearization();
F_true = [eye(2)+A*h.dt_c, zeros(2,1); 0 0 1]; G_true = [B*h.dt_c; 0];
P_star = iadp.solveDiscountedRiccati(F_true, G_true, h.Q_aug, h.options.R, h.options.gamma);
[KX_star, KdX_star, Kd_star] = iadp.optimalGain(F_true, G_true, P_star, h.options.R, h.options.gamma);

% --- learned kernel & gain (identified model + online-learned P) ---
Ff_aug = [h.F_final, zeros(2,1); 0 0 1]; Gf_aug = [h.G_final; 0];
[KX_l, KdX_l, Kd_l] = iadp.optimalGain(Ff_aug, Gf_aug, h.P_final, h.options.R, h.options.gamma);

fprintf('Learned vs optimal feedback gain on X=[q,alpha,qref]:\n');
fprintf('  K_X learned : [%+.3f %+.3f %+.3f]\n', KX_l);
fprintf('  K_X optimal : [%+.3f %+.3f %+.3f]\n', KX_star);

% --- ONLINE convergence: P snapshots logged at every 20 Hz backup of the run ---
dP = zeros(h.n_backups,1);
for i = 1:h.n_backups
    dP(i) = norm(h.P_log(:,:,i) - P_star,'fro')/norm(P_star,'fro');
end

f = figure('Position',[100 100 800 350],'Color','w');
n_show = min(50, h.n_backups);
semilogy(1:n_show, dP(1:n_show), 'b-o','LineWidth',1.2,'MarkerSize',3); hold on;
yline(dP(end),'r--',sprintf('%.1f%% floor (model error)',100*dP(end)));
grid on;
xlabel('policy-evaluation backup (20 Hz, online; first 50 of 420 shown, remainder stays at floor)');
ylabel('||P_i - P*|| / ||P*||');
title('Online value iteration converging to the LQR-optimal kernel');
exportgraphics(f, [figdir 'exp2_lqr_convergence.png'], 'Resolution', 150);

save([resdir 'exp2_lqr.mat'], 'P_star','KX_star','KdX_star','Kd_star', ...
     'KX_l','KdX_l','Kd_l','dP');
fprintf('Exp2: dP(1)=%.3f -> dP(end)=%.4f over %d online backups\n', dP(1), dP(end), h.n_backups);

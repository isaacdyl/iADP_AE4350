%% Step 3 verification: discounted Riccati benchmark on the augmented system
clear; clc;
addpath('D:\Bio_inspired\iADP_Control_Assignment');

dt = 1e-3; gamma = 0.9; Q = 400; R = 1;
C_E = [1, 0, -1];

plant = iadp.ShortPeriodAircraft(dt); plant.reset();   % trim: q=alpha=0
[A,B] = plant.linearization();
F = eye(2) + A*dt;
G = B*dt;

% Augment with a 'frozen reference' row/col: qref_next ~= qref_t (held constant
% for the purpose of this benchmark -- standard quasi-static assumption for an
% LQ tracking gain; the real reference is exogenous/time-varying in the sim).
F_aug = [F, zeros(2,1); 0, 0, 1];
G_aug = [G; 0];
Q_aug = Q * (C_E' * C_E);

P_star = iadp.solveDiscountedRiccati(F_aug, G_aug, Q_aug, R, gamma);
disp('P_star ='); disp(P_star);
fprintf('eig(P_star): '); disp(eig(P_star)');
fprintf('All eigenvalues positive (PD)? %d\n', all(eig(P_star) > 0));

[K_X, K_dX, K_d] = iadp.optimalGain(F_aug, G_aug, P_star, R, gamma);
disp('K_X ='); disp(K_X);
disp('K_dX ='); disp(K_dX);
disp('K_d ='); disp(K_d);

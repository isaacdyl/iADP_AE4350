%% Isolate RLS correctness on a trivial static regression (no plant in the loop)
clear; clc;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
rng(0);

n_w = 3; n_y = 2;
Theta_true = [0.5 -0.2; 0.1 0.9; -3.0 0.05];  % (n_w x n_y)
rls = iadp.IncrementalRLS(n_y, 1, 1.0, 50);  % n_x=2 (->n_y), n_u=1 -> n_w=3

N = 2000;
err = zeros(N,1);
for k = 1:N
    w = randn(n_w,1);
    y = Theta_true' * w + 1e-4*randn(n_y,1);
    y_pred = rls.predict(w);
    rls.update(y, y_pred, w);
    err(k) = norm(rls.Theta - Theta_true, 'fro') / norm(Theta_true,'fro');
end
idx = [1 5 10 20 50 100 500 1000 2000];
for ii = idx
    fprintf('k=%4d  relerr=%.5f\n', ii, err(ii));
end
disp('Theta final:'); disp(rls.Theta);
disp('Theta true:'); disp(Theta_true);

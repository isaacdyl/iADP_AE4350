%% Step 1 verification: open-loop excitation + batch OLS recovers true F,G
clear; clc;
addpath('D:\Bio_inspired\iADP_Control_Assignment');

dt = 1e-3;
T = 10.0;
n = round(T/dt);
t = (0:n-1)'*dt;

plant = iadp.ShortPeriodAircraft(dt);
plant.reset();

freqs = 2*pi*[0.7, 1.3, 2.1, 3.4];
delta_hist = zeros(n,1);
q_hist = zeros(n,1);
a_hist = zeros(n,1);

for k = 1:n
    s = 0;
    for i = 1:numel(freqs)
        s = s + sin(freqs(i)*t(k) + (i-1));
    end
    delta_cmd = deg2rad(2.0) * (s/numel(freqs));
    [xnext, delta_true] = plant.step(delta_cmd);
    delta_hist(k) = delta_true;
    q_hist(k) = xnext(1);
    a_hist(k) = xnext(2);
end

X = [q_hist, a_hist];
dX = diff(X);
dX_t = dX(1:end-1,:);
dX_next = dX(2:end,:);
ddelta = diff(delta_hist);
ddelta = ddelta(1:end-1);

W = [dX_t, ddelta];
Theta = W \ dX_next;   % least squares, Theta is (3x2): rows [F_row1;F_row2;G_row]
F_fit = Theta(1:2,:)';
G_fit = Theta(3,:)';

[A,B] = plant.linearization();
F_true_approx = eye(2) + A*dt;
G_true_approx = B*dt;

disp('F_fit ='); disp(F_fit);
disp('F_true_approx ='); disp(F_true_approx);
disp('G_fit ='); disp(G_fit);
disp('G_true_approx ='); disp(G_true_approx);
fprintf('delta range (deg): [%.3f, %.3f]\n', rad2deg(min(delta_hist)), rad2deg(max(delta_hist)));
fprintf('Max abs error F: %.3e | Max abs error G: %.3e\n', ...
        max(abs(F_fit(:)-F_true_approx(:))), max(abs(G_fit(:)-G_true_approx(:))));

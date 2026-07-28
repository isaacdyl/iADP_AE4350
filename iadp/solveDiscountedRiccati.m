function P = solveDiscountedRiccati(F, G, Q, R, gamma, n_iter, tol)
    % Benchmark: iterates the same discounted Riccati fixed point as the online
    % critic, but on the given (usually true) F,G, giving the P* to compare
    % the learned kernel against.
    if nargin < 6 || isempty(n_iter), n_iter = 5000; end
    if nargin < 7 || isempty(tol), tol = 1e-12; end

    n = size(F,1);
    P = eye(n);
    for i = 1:n_iter
        M = R + gamma * (G' * P * G);
        FtPG = F' * P * G;
        P_new = Q + gamma * (F' * P * F) - gamma^2 * FtPG * (M \ (G' * P * F));
        P_new = 0.5 * (P_new + P_new');
        if max(abs(P_new(:) - P(:))) < tol
            P = P_new;
            break
        end
        P = P_new;
    end
end

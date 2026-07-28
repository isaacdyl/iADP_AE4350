classdef KernelPolicyEvaluator < handle
    % Critic for the quadratic value fn V(X) = X'*P*X.
    %
    % Default path is bellmanBackup(): one discounted-Riccati value-iteration
    % step per critic tick on the identified (F,G). Contraction for gamma<1,
    % so P converges to the optimum of the identified model and stays PSD.
    %
    % addSample()/solveBuffer() keep the data-driven Kronecker LS solve from
    % the reference papers, for comparison. Note the tracking cost here only
    % penalises q-q_ref, so the true kernel is near-singular along the
    % uncosted alpha direction and sensor noise flips that eigenvalue negative
    % on most LS windows -> solved kernels get projected back onto the PSD cone.

    properties
        n_X
        gamma
        buffer_size
        reg
        eig_floor = 1e-6;     % PSD projection floor
        cond_max = 1e12;      % skip solve if normal matrix worse than this
        P
        A_buf;
        b_buf;
        n_buf;
        n_solved = 0;         % accepted solves
        n_skipped = 0;        % skipped (ill-conditioned)
        n_projected = 0;      % needed PSD projection
    end

    methods
        function obj = KernelPolicyEvaluator(n_X, gamma, buffer_size, reg, P0)
            obj.n_X = n_X;
            obj.gamma = gamma;
            obj.buffer_size = buffer_size;
            if nargin < 4 || isempty(reg), reg = 1e-6; end
            obj.reg = reg;
            if nargin < 5 || isempty(P0)
                obj.P = eye(n_X);
            else
                obj.P = P0;
            end
            obj.A_buf = zeros(buffer_size, n_X^2);
            obj.b_buf = zeros(buffer_size, 1);
            obj.n_buf = 0;
        end

        function addSample(obj, X_t, X_next, cost_t)
            V_t = cost_t + obj.gamma * (X_next' * obj.P * X_next);
            obj.n_buf = obj.n_buf + 1;
            obj.A_buf(obj.n_buf, :) = kron(X_t', X_t');
            obj.b_buf(obj.n_buf) = V_t;
            if obj.n_buf >= obj.buffer_size
                obj.solveBuffer();
                obj.n_buf = 0;
            end
        end

        function solveBuffer(obj)
            A = obj.A_buf(1:obj.n_buf, :);
            b = obj.b_buf(1:obj.n_buf);
            n2 = size(A,2);
            N = A'*A + obj.reg*eye(n2);
            if rcond(N) < 1/obj.cond_max
                obj.n_skipped = obj.n_skipped + 1;
                return;   % too ill-conditioned this window, keep prev P
            end
            vecP = N \ (A'*b);
            P_new = reshape(vecP, obj.n_X, obj.n_X);
            P_new = 0.5*(P_new + P_new');
            obj.P = obj.projectPSD(P_new);
            obj.n_solved = obj.n_solved + 1;
        end

        function bellmanBackup(obj, F, G, Q, R)
            % one discounted-Riccati VI step on the current learned model
            g = obj.gamma;
            P = obj.P;
            M = R + g*(G'*P*G);
            FtPG = F'*P*G;
            P_new = Q + g*(F'*P*F) - g^2*(FtPG*(M\(G'*P*F)));
            P_new = 0.5*(P_new + P_new');
            obj.P = obj.projectPSD(P_new);
            obj.n_solved = obj.n_solved + 1;
        end

        function P_new = projectPSD(obj, P_new)
            % clip eigenvalues to a small positive floor
            [V,D] = eig(P_new); d = diag(D);
            if any(d < obj.eig_floor)
                d = max(d, obj.eig_floor);
                P_new = V*diag(d)*V'; P_new = 0.5*(P_new + P_new');
                obj.n_projected = obj.n_projected + 1;
            end
        end

        function reset(obj, P0)
            if nargin < 2 || isempty(P0)
                obj.P = eye(obj.n_X);
            else
                obj.P = P0;
            end
            obj.n_buf = 0;
            obj.n_solved = 0; obj.n_skipped = 0; obj.n_projected = 0;
        end
    end
end

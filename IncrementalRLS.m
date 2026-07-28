classdef IncrementalRLS < handle
    % RLS identification of the incremental model
    %   dX_{t+1} ~= F_tilde*dX_t + G_tilde*ddelta_t
    % Standard recursion with forgetting factor. Cold-started with a large
    % initial covariance (Phi0 = cov0*I), which for the first samples behaves
    % close to a batch LS fit.

    properties
        n_x
        n_u
        n_w
        gamma_rls
        Theta
        Phi
    end

    methods
        function obj = IncrementalRLS(n_x, n_u, gamma_rls, cov0)
            if nargin < 3 || isempty(gamma_rls), gamma_rls = 0.999; end
            if nargin < 4 || isempty(cov0), cov0 = 1e8; end
            obj.n_x = n_x;
            obj.n_u = n_u;
            obj.n_w = n_x + n_u;
            obj.gamma_rls = gamma_rls;
            obj.reset(cov0);
        end

        function reset(obj, cov0)
            if nargin < 2 || isempty(cov0), cov0 = 1e8; end
            obj.Theta = zeros(obj.n_w, obj.n_x);
            obj.Phi = eye(obj.n_w) * cov0;
        end

        function update(obj, dX_meas, dX_pred, W)
            eps = dX_meas - dX_pred;              % innovation (n_x,1)
            PhiW = obj.Phi * W;
            denom = obj.gamma_rls + W' * PhiW;
            gain = PhiW / denom;

            obj.Theta = obj.Theta + gain * eps';
            obj.Phi = (obj.Phi - (PhiW * PhiW') / denom) / obj.gamma_rls;
            obj.Phi = 0.5 * (obj.Phi + obj.Phi');  % keep symmetric
        end

        function dX_pred = predict(obj, W)
            dX_pred = obj.Theta' * W;
        end

        function [F_tilde, G_tilde] = FG(obj)
            F_tilde = obj.Theta(1:obj.n_x, :)';
            G_tilde = obj.Theta(obj.n_x+1:end, :)';
        end
    end
end

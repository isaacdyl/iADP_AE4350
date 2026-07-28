function [K_X, K_dX, K_d] = optimalGain(F, G, P, R, gamma)
    % Feedback gains implied by the closed-form policy, for learned-vs-true
    % comparison:  ddelta = -K_X*X_t - K_dX*dX_t - K_d*delta_prev
    GtP = G' * P;
    M = R + gamma * (GtP * G);
    K_X = gamma * (M \ GtP);
    K_dX = gamma * (M \ (GtP * F));
    K_d = M \ R;
end

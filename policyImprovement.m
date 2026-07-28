function dDelta = policyImprovement(Ftil, Gtil, P, X_t, dX_t, delta_prev, R, gamma)
    % Closed-form incremental control law ( eq. 11):
    %   ddelta = -(R + g*G'PG)^-1 * [R*delta_prev + g*G'P*X_t + g*G'P*F*dX_t]
    GtP = Gtil' * P;
    M = R + gamma * (GtP * Gtil);
    rhs = R*delta_prev + gamma*(GtP*X_t) + gamma*(GtP*Ftil*dX_t);
    dDelta = -(M \ rhs);
end

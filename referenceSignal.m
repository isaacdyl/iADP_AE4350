function qref = referenceSignal(t, varargin)
    % Repeating 3-2-1-1 multistep pitch-rate command.
    p = inputParser;
    addParameter(p, 'period', 8.0);
    addParameter(p, 'amp', deg2rad(5.0));
    addParameter(p, 'unit', 0.7);
    addParameter(p, 't0', 1.0);
    addParameter(p, 'n_repeats', 6);
    parse(p, varargin{:});
    period = p.Results.period; amp = p.Results.amp;
    unit = p.Results.unit; t0 = p.Results.t0; n_repeats = p.Results.n_repeats;

    qref = zeros(size(t));
    for k = 0:(n_repeats-1)
        tau = t - (t0 + k*period);
        seg = zeros(size(t));
        seg(tau >= 0      & tau < 3*unit) = amp;
        seg(tau >= 3*unit & tau < 5*unit) = -amp;
        seg(tau >= 5*unit & tau < 6*unit) = amp;
        seg(tau >= 6*unit & tau < 7*unit) = -amp;
        qref = qref + seg;
    end
end

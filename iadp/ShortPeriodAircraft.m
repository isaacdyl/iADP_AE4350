classdef ShortPeriodAircraft < handle
    % Longitudinal short-period truth plant: states [q; alpha], input delta_e.
    % Linear short-period derivatives + nonlinear pitching-moment term
    % Cm_alpha2*alpha*|alpha|, plus a 1st-order rate/position-limited actuator.

    properties
        dt = 1e-3

        % short-period derivatives (representative business-jet, cruise)
        Zalpha = -1.40
        Zq = 1.0
        Zde = -0.05
        Malpha = -2.6
        Mq = -0.65
        Mde = -5.2
        Cm_alpha2 = -3.0

        % actuator
        tau_act = 1/15
        rate_limit = deg2rad(60.0)
        pos_limit = deg2rad(25.0)

        x = [0; 0]
        delta_true = 0.0
        fault_factor = 1.0
    end

    properties (Dependent)
        q
        alpha
    end

    methods
        function obj = ShortPeriodAircraft(dt)
            if nargin >= 1 && ~isempty(dt)
                obj.dt = dt;
            end
        end

        function reset(obj, x0)
            if nargin < 2 || isempty(x0)
                obj.x = [0; 0];
            else
                obj.x = x0(:);
            end
            obj.delta_true = 0.0;
            obj.fault_factor = 1.0;
        end

        function injectFault(obj, factor)
            % 1.0 healthy, 0<f<1 effectiveness loss, f<0 sign reversal
            obj.fault_factor = factor;
        end

        function setMalpha(obj, value)
            % simulates a cg shift (static margin change)
            obj.Malpha = value;
        end

        function xdot = xDot(obj, x, delta)
            q = x(1); alpha = x(2);
            alpha_dot = obj.Zalpha*alpha + obj.Zq*q + obj.Zde*delta;
            Mde_eff = obj.Mde * obj.fault_factor;
            q_dot = obj.Malpha*alpha + obj.Mq*q + Mde_eff*delta ...
                    + obj.Cm_alpha2*alpha*abs(alpha);
            xdot = [q_dot; alpha_dot];
        end

        function delta_new = actuatorStep(obj, delta_cmd)
            rate = (delta_cmd - obj.delta_true) / obj.tau_act;
            rate = max(min(rate, obj.rate_limit), -obj.rate_limit);
            delta_new = obj.delta_true + rate * obj.dt;
            delta_new = max(min(delta_new, obj.pos_limit), -obj.pos_limit);
        end

        function [xnext, delta_true] = step(obj, delta_cmd)
            % actuator (Euler) then plant (RK4), one 1 ms step
            obj.delta_true = obj.actuatorStep(delta_cmd);
            delta = obj.delta_true;
            x0 = obj.x;
            dt = obj.dt;
            k1 = obj.xDot(x0, delta);
            k2 = obj.xDot(x0 + dt/2*k1, delta);
            k3 = obj.xDot(x0 + dt/2*k2, delta);
            k4 = obj.xDot(x0 + dt*k3, delta);
            obj.x = x0 + dt/6*(k1 + 2*k2 + 2*k3 + k4);
            xnext = obj.x;
            delta_true = obj.delta_true;
        end

        function [A, B] = linearization(obj)
            % local Jacobian at the current state (incl. fault factor)
            alpha = obj.x(2);
            dMda = obj.Malpha + 2*obj.Cm_alpha2*abs(alpha);
            A = [obj.Mq, dMda; obj.Zq, obj.Zalpha];
            B = [obj.Mde*obj.fault_factor; obj.Zde];
        end

        function val = get.q(obj)
            val = obj.x(1);
        end

        function val = get.alpha(obj)
            val = obj.x(2);
        end
    end
end

%% Experiment 4: Sensitivity analysis (statistics over seeds)
clear; clc; close all;
addpath('D:\Bio_inspired\iADP_Control_Assignment');
figdir = 'D:\Bio_inspired\iADP_Control_Assignment\figures\';
resdir = 'D:\Bio_inspired\iADP_Control_Assignment\results\';
dt = 1e-3; seeds = 1:8; T = 20.0;

% settled-tracking RMSE metric (excludes step transients)
settledRMSE = @(h) local_settled(h, dt);

sweeps = struct();
sweeps.gamma          = [0.80 0.85 0.90 0.95 0.97];
sweeps.QR             = [25 100 400 1600 6400];   % Q with R=1
sweeps.learn_duration = [1.0 1.5 2.0 3.0 4.0];
sweeps.excite_amp     = [0.5 1.0 1.5 2.0 3.0];
names = fieldnames(sweeps);

f = figure('Position',[80 80 1000 700],'Color','w');
for s = 1:numel(names)
    nm = names{s}; vals = sweeps.(nm);
    M = zeros(numel(vals), numel(seeds));
    for i = 1:numel(vals)
        for j = 1:numel(seeds)
            args = {'T_total',T,'seed',seeds(j)};
            switch nm
                case 'gamma',      args = [args {'gamma',vals(i)}];
                case 'QR',         args = [args {'Q',vals(i),'R',1}];
                case 'learn_duration', args = [args {'learn_duration',vals(i)}];
                case 'excite_amp', args = [args {'excite_amp_deg',vals(i)}];
            end
            h = iadp.runSimulation(args{:});
            M(i,j) = settledRMSE(h);
        end
    end
    subplot(2,2,s);
    mu = mean(M,2); sd = std(M,0,2);
    errorbar(1:numel(vals), mu, sd, 'o-','LineWidth',1.3,'MarkerFaceColor','b'); grid on;
    set(gca,'XTick',1:numel(vals),'XTickLabel',string(vals));
    xlabel(nm,'Interpreter','none'); ylabel('settled RMSE [deg/s]');
    title(sprintf('Sensitivity to %s (mean\\pm std, %d seeds)', nm, numel(seeds)),'Interpreter','tex');
    save([resdir 'exp4_' nm '.mat'], 'vals','M');
end
exportgraphics(f, [figdir 'exp4_sensitivity.png'], 'Resolution', 150);
fprintf('Exp4 done. Figures + per-sweep .mat saved.\n');

function r = local_settled(h, dt)
    qref = h.qref; q = h.q; n_warm = round(h.options.learn_duration/dt);
    dref = [0; diff(qref)]; seg_start = find(dref~=0);
    err = [];
    for i = 1:numel(seg_start)-1
        a = seg_start(i); b = seg_start(i+1)-1;
        if b-a > 200 && a > n_warm
            tail = round(a + 0.6*(b-a)):b; err = [err; q(tail)-qref(tail)]; %#ok<AGROW>
        end
    end
    if isempty(err), r = NaN; else, r = rad2deg(sqrt(mean(err.^2))); end
end

% Fit COMSOL office-space admittance tables to EDG AbsorbBC coefficients.
%
% The input tables provide the frequency samples. The target admittance is
% evaluated from the partial-fraction functions pff1/pff2/pff3 recovered from
% the MPH, because those functions are what COMSOL's active impedance features
% actually reference. EDG AbsorbBC expects a rational approximation of
% R = (1 - rho0*c0*Y) / (1 + rho0*c0*Y).

function fit_office_space_admittance()
close all;

script_path = which(mfilename());
case_dir = fileparts(script_path);
if isempty(case_dir)
  case_dir = pwd;
end
case_candidates = {
  case_dir;
  fullfile(pwd, 'examples', 'office_space_admittance_carpet');
  fullfile(fileparts(pwd), 'edg-acoustics', 'examples', 'office_space_admittance_carpet');
};
for candidate_index = 1:length(case_candidates)
  candidate = case_candidates{candidate_index};
  if exist(fullfile(candidate, 'office_space_admittance_carpet.txt'), 'file')
    case_dir = candidate;
    break;
  end
end
repo_fit_dir = fullfile(case_dir, '..', 'material_fit');
addpath(repo_fit_dir);

rho0 = 1.2;
c0 = 343.0;
z0 = rho0 * c0;
freq_max_passivity = 2000.0;
n_iter = 20;

materials = {
  'carpet',  'office_space_admittance_carpet.txt',   8, 5.0e-4, ...
    0.0012444627411150688, ...
    [-5.585194776664902], [-4497.290543265215], [], [];
  'ceiling', 'office_space_admittance_ceiling.txt', 16, 5.0e-4, ...
    0.0011936661364735238, ...
    [-0.07059007074122078, -0.015066325082052768, -0.0027045344243451547], ...
    [-356.0149965208861, -71.55555638529185, -18.8870439151275], ...
    [1.57909398381373+3.057098824668871i, ...
     0.016418139638804598+0.022281900876605622i, ...
     -0.005924694102047328+0.02841509415752516i, ...
     -0.025007907070339248+0.012448132996876011i], ...
    [-1571.2965232398606+3330.8847903420397i, ...
     -48.32149963294501+1674.6385380543861i, ...
     -42.88340489570168+1109.2589799576533i, ...
     -35.01515188585684+549.2945320321779i];
  'gypsum',  'office_space_admittance_gypsum.txt',  16, 2.0e-3, ...
    1.5195323452478004e-7, ...
    [-0.0021839404783796828], [-206.3653326198353], ...
    [0.09343613755922382+0.0031239357750316014i, ...
     0.025175646224803287+0.006093240245722001i], ...
    [-123.3308289114408+3301.441711616325i, ...
     -20.11371620520235+66.79104203470644i];
};

for material_index = 1:size(materials, 1)
  name = materials{material_index, 1};
  input_name = materials{material_index, 2};
  n_poles = materials{material_index, 3};
  rms_limit = materials{material_index, 4};
  pffYInf = materials{material_index, 5};
  pffR = materials{material_index, 6};
  pffXi = materials{material_index, 7};
  pffQ = materials{material_index, 8};
  pffZeta = materials{material_index, 9};
  input_path = fullfile(case_dir, input_name);

  data = dlmread(input_path, '', 1, 0);
  freq = data(:, 1).';
  rawAdmittance = data(:, 2).' + 1i * data(:, 3).';
  keep = freq > 0;
  freq = freq(keep);
  rawAdmittance = rawAdmittance(keep);

  comsolPffAdmittance = eval_comsol_pff(freq, pffYInf, pffR, pffXi, pffQ, pffZeta);

  omega = 2 * pi * freq;
  s = 1i * omega;
  target = (1 - z0 * comsolPffAdmittance) ./ (1 + z0 * comsolPffAdmittance);
  rawTableReflection = (1 - z0 * rawAdmittance) ./ (1 + z0 * rawAdmittance);
  table_pff_reflection_rms = sqrt(mean(abs(rawTableReflection - target).^2));
  ns = length(freq);

  opts.asymp = 2;
  opts.cmplx_ss = 1;
  opts.relax = 1;
  opts.stable = 1;
  opts.skip_pole = 0;
  opts.skip_res = 1;
  opts.spy1 = 0;
  opts.spy2 = 0;
  opts.logx = 0;
  opts.logy = 1;
  opts.errplot = 0;
  opts.phaseplot = 0;
  opts.legend = 0;

  pole_freq = logspace(log10(max(min(freq), 1.0)), log10(max(freq)), n_poles);
  poles = -2 * pi * pole_freq;
  weight = ones(1, ns);

  fprintf('Fitting %s with %d poles\n', name, n_poles);
  for iter = 1:n_iter
    if iter == n_iter
      opts.skip_res = 0;
    end
    [SER, poles, rmserr, fit] = vectfit3(target, s, poles, weight, opts);
  end

  A = diag(full(SER.A));
  C = SER.C(:).';
  RI = real(SER.D(1));

  AS = [];
  lambdaS = [];
  BS = [];
  CS = [];
  alphaS = [];
  betaS = [];

  for pole_index = 1:length(A)
    pole = A(pole_index);
    residue = C(pole_index);
    if abs(imag(pole)) < 1.0e-8
      lambdaS(end + 1) = -real(pole);
      AS(end + 1) = real(residue);
    elseif imag(pole) > 0
      alphaS(end + 1) = -real(pole);
      betaS(end + 1) = imag(pole);
      BS(end + 1) = 2 * real(residue);
      CS(end + 1) = -2 * imag(residue);
    end
  end

  scale = find_passive_scale(RI, AS, lambdaS, BS, CS, alphaS, betaS, freq_max_passivity);
  AS = AS * scale;
  BS = BS * scale;
  CS = CS * scale;

  fit_edg = eval_edg_reflection(omega, RI, AS, lambdaS, BS, CS, alphaS, betaS);
  rms_error = sqrt(mean(abs(fit_edg - target).^2));
  max_error = max(abs(fit_edg - target));
  omega_check = linspace(1, 2 * pi * freq_max_passivity, 5000);
  max_abs_R = max(abs(eval_edg_reflection(omega_check, RI, AS, lambdaS, BS, CS, alphaS, betaS)));

  fprintf('  rms=%g max_err=%g max_abs_R=%g scale=%g table_pff_rms=%g\n', ...
          rms_error, max_error, max_abs_R, scale, table_pff_reflection_rms);
  if rms_error > rms_limit
    error('%s fit RMS %g exceeds limit %g', name, rms_error, rms_limit);
  end
  if max_abs_R > 1.0 + 1.0e-8
    error('%s fit is not passive: max |R| = %g', name, max_abs_R);
  end

  ApproxValue = fit_edg;
  trueValue = target;
  target_source = 'COMSOL partial-fraction admittance';
  output_path = fullfile(case_dir, [name '.mat']);
  save('-mat', output_path, 'RI', 'AS', 'lambdaS', 'BS', 'CS', ...
       'alphaS', 'betaS', 'freq', 'ApproxValue', 'trueValue', ...
       'rawAdmittance', 'comsolPffAdmittance', 'rawTableReflection', ...
       'pffYInf', 'pffR', 'pffXi', 'pffQ', 'pffZeta', 'target_source', ...
       'rms_error', 'max_error', 'max_abs_R', 'table_pff_reflection_rms');

  fig = figure('visible', 'off');
  subplot(2, 1, 1);
  semilogx(freq, abs(trueValue), 'k-', freq, abs(ApproxValue), 'r--');
  grid on;
  ylabel('|R|');
  legend('COMSOL PFF', 'EDG fit');
  title([name ' reflection fit']);
  subplot(2, 1, 2);
  semilogx(freq, unwrap(angle(trueValue)), 'k-', freq, unwrap(angle(ApproxValue)), 'r--');
  grid on;
  xlabel('Frequency (Hz)');
  ylabel('phase(R) [rad]');
  print(fig, fullfile(case_dir, [name '_fit_diagnostics.png']), '-dpng', '-r180');
  close(fig);
end
end

function value = eval_comsol_pff(freq, YInf, R, xi, Q, zeta)
  comsol_s = 1i * freq;
  value = YInf * ones(size(freq));
  for k = 1:length(R)
    value = value + R(k) ./ (comsol_s - xi(k));
  end
  for k = 1:length(Q)
    value = value + 0.5 * ( ...
      Q(k) ./ (comsol_s - zeta(k)) + ...
      conj(Q(k)) ./ (comsol_s - conj(zeta(k))));
  end
end

function value = eval_edg_reflection(omega, RI, AS, lambdaS, BS, CS, alphaS, betaS)
  value = RI * ones(size(omega));
  for k = 1:length(AS)
    value = value + AS(k) ./ (1i * omega + lambdaS(k));
  end
  for k = 1:length(BS)
    value = value + 0.5 * ( ...
      (BS(k) + 1i * CS(k)) ./ (alphaS(k) + 1i * betaS(k) + 1i * omega) + ...
      (BS(k) - 1i * CS(k)) ./ (alphaS(k) - 1i * betaS(k) + 1i * omega));
  end
end

function scale = find_passive_scale(RI, AS, lambdaS, BS, CS, alphaS, betaS, freq_max)
  omega_check = linspace(1, 2 * pi * freq_max, 5000);
  current = max(abs(eval_edg_reflection(omega_check, RI, AS, lambdaS, BS, CS, alphaS, betaS)));
  if current <= 1.0
    scale = 1.0;
    return;
  end
  lo = 0.0;
  hi = 1.0;
  for iter = 1:40
    mid = 0.5 * (lo + hi);
    trial = max(abs(eval_edg_reflection(omega_check, RI, AS * mid, lambdaS, ...
                                        BS * mid, CS * mid, alphaS, betaS)));
    if trial <= 1.0
      lo = mid;
    else
      hi = mid;
    end
  end
  scale = lo;
end

fit_office_space_admittance();

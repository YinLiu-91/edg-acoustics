function fit_er_material(compressibility_file, density_file, output_file)
%FIT_ER_MATERIAL Fit ER porous material beta/rho tables with vectfit3.

  if nargin < 1
    compressibility_file = fullfile(fileparts(mfilename('fullpath')), ...
      'porous_absorber_time_domain_compressibility.zh_CN.txt');
  end
  if nargin < 2
    density_file = fullfile(fileparts(mfilename('fullpath')), ...
      'porous_absorber_time_domain_density.zh_CN.txt');
  end
  if nargin < 3
    output_file = fullfile(fileparts(mfilename('fullpath')), 'er_material_fit.mat');
  end

  case_dir = fileparts(mfilename('fullpath'));
  addpath(fullfile(case_dir, '..', '..', 'material_fit'));

  [freq_beta, beta_samples] = read_complex_table(compressibility_file);
  [freq_rho, rho_samples] = read_complex_table(density_file);

  if any(freq_beta ~= freq_rho)
    error('compressibility and density frequency grids do not match');
  end

  n_poles = 8;
  n_iter = 20;
  [A_beta, B_beta, C_beta, D_beta, rmserr_beta, fit_beta] = ...
    fit_response(freq_beta, beta_samples, n_poles, n_iter);
  [A_rho, B_rho, C_rho, D_rho, rmserr_rho, fit_rho] = ...
    fit_response(freq_rho, rho_samples, n_poles, n_iter);

  freq = freq_beta;
  beta_samples_real = real(beta_samples);
  beta_samples_imag = imag(beta_samples);
  rho_samples_real = real(rho_samples);
  rho_samples_imag = imag(rho_samples);
  fit_beta_real = real(fit_beta);
  fit_beta_imag = imag(fit_beta);
  fit_rho_real = real(fit_rho);
  fit_rho_imag = imag(fit_rho);

  save('-mat7-binary', output_file, ...
    'A_beta', 'B_beta', 'C_beta', 'D_beta', 'rmserr_beta', ...
    'A_rho', 'B_rho', 'C_rho', 'D_rho', 'rmserr_rho', ...
    'freq', ...
    'beta_samples_real', 'beta_samples_imag', ...
    'rho_samples_real', 'rho_samples_imag', ...
    'fit_beta_real', 'fit_beta_imag', ...
    'fit_rho_real', 'fit_rho_imag');
end

function [freq, values] = read_complex_table(filename)
  fid = fopen(filename, 'r');
  if fid < 0
    error(['failed to open file: ', filename]);
  end
  cleaner = onCleanup(@() fclose(fid));
  rows = textscan(fid, '%f%f%f', 'CommentStyle', '%', 'CollectOutput', true);
  data = rows{1};
  freq = data(:, 1);
  values = data(:, 2) + 1i * data(:, 3);
end

function [A, B, C, D, rmserr, fit_values] = fit_response(freq, values, n_poles, n_iter)
  omega = 2 * pi * freq(:).';
  s = 1i * omega;
  response = values(:).';
  poles = -logspace(log10(min(omega)), log10(max(omega)), n_poles);
  weight = ones(size(s));

  opts.relax = 1;
  opts.stable = 1;
  opts.skip_pole = 0;
  opts.skip_res = 1;
  opts.spy1 = 0;
  opts.spy2 = 0;
  opts.logx = 0;
  opts.logy = 0;
  opts.errplot = 0;
  opts.phaseplot = 0;
  opts.legend = 0;
  opts.asymp = 2;
  opts.cmplx_ss = 0;

  for iter = 1:n_iter
    if iter == n_iter
      opts.skip_res = 0;
    end
    [SER, poles, rmserr, fit_values] = vectfit3(response, s, poles, weight, opts); %#ok<ASGLU>
  end

  if isfield(SER, 'E') && max(abs(SER.E(:))) > 1e-8
    error('Expected zero E term from vectfit3 asymp=2 fit.');
  end

  A = real(full(SER.A));
  B = real(full(SER.B));
  C = real(full(SER.C));
  D = real(full(SER.D));
  fit_values = fit_values(:);
end

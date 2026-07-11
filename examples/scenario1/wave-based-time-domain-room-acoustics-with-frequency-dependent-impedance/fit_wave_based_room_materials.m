function fit_wave_based_room_materials()
% Fit COMSOL admittance tables into EDG Acoustics boundary .mat files.
%
% The input text files contain normal-incidence admittance Y. EDG Acoustics
% expects a vector-fitted reflection coefficient R, so this script converts
% Y to R before calling the repository's vectfit3.m.

script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

addpath(fullfile(script_dir, '..', '..', 'material_fit'));

rho0 = 1.213;
c0 = 343.0;
z0 = rho0 * c0;
niter = 20;

materials = {
    'carpet',  'wave_based_room_admittance_carpet.zh_CN.txt',  5, false;
    'ceiling', 'wave_based_room_admittance_ceiling.zh_CN.txt', 5, false;
    'sofa',    'wave_based_room_admittance_sofa.zh_CN.txt',    5, true;
    'walls',   'wave_based_room_admittance_wall.zh_CN.txt',    5, false;
};

for row = 1:size(materials, 1)
    name = materials{row, 1};
    filename = fullfile(script_dir, materials{row, 2});
    npoles = materials{row, 3};
    remove_real_poles = materials{row, 4};
    fit_one_material(name, filename, npoles, remove_real_poles, niter, z0, script_dir);
end
end


function fit_one_material(name, filename, npoles, remove_real_poles, niter, z0, out_dir)
data = load_admittance_table(filename);
freq = data(:, 1).';
Y = data(:, 2).' + 1i * data(:, 3).';
reflection = (1 - z0 * Y) ./ (1 + z0 * Y);

omega = 2 * pi * freq;
s = 1i * omega;
poles = -2 * pi * linspace(freq(1), freq(end), npoles);
weight = ones(1, length(freq));

opts.relax = 1;
opts.stable = 1;
opts.asymp = 1;
opts.cmplx_ss = 1;
opts.skip_pole = 0;
opts.skip_res = 1;
opts.spy1 = 0;
opts.spy2 = 0;
opts.logx = 0;
opts.logy = 1;
opts.errplot = 0;
opts.phaseplot = 0;
opts.legend = 0;

fprintf('Fitting %s with %d poles...\n', name, npoles);
for iter = 1:niter
    if iter == niter
        opts.skip_res = 0;
    end
    [SER, poles, rmserr, fit] = vectfit3(reflection, s, poles, weight, opts);
end

[AS, lambdaS, BS, CS, alphaS, betaS] = extract_edg_coefficients(SER);
if remove_real_poles
    AS = [];
    lambdaS = [];
end

ApproxValue = fit;
TrueFun = reflection;
rms_error = rmserr;
out_file = fullfile(out_dir, sprintf('%s_N%d_Fmax%d.mat', name, npoles, round(freq(end))));

save('-mat', out_file, ...
    'AS', 'lambdaS', 'BS', 'CS', 'alphaS', 'betaS', ...
    'ApproxValue', 'TrueFun', 'freq', 'rms_error');
fprintf('Wrote %s\n', out_file);
end


function data = load_admittance_table(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('Could not open %s', filename);
end

rows = [];
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    line = strtrim(line);
    if isempty(line) || line(1) == '%'
        continue;
    end
    values = sscanf(line, '%f');
    if numel(values) >= 3
        if numel(values) == 3
            values(4) = NaN;
        end
        rows = [rows; values(1:4).']; %#ok<AGROW>
    end
end
fclose(fid);

if isempty(rows)
    error('No numeric admittance rows found in %s', filename);
end
data = rows;
end


function [AS, lambdaS, BS, CS, alphaS, betaS] = extract_edg_coefficients(SER)
A = full(SER.A);
C = SER.C;
diagA = diag(A);

AS = [];
lambdaS = [];
BS = [];
CS = [];
alphaS = [];
betaS = [];

used = false(length(diagA), 1);
tol = 1e-8;

for idx = 1:length(diagA)
    if used(idx)
        continue;
    end

    pole = diagA(idx);
    residue = C(idx);
    if abs(imag(pole)) < tol
        lambdaS(end + 1) = -real(pole); %#ok<AGROW>
        AS(end + 1) = real(residue); %#ok<AGROW>
        used(idx) = true;
    else
        pair = find(~used & (abs(diagA - conj(pole)) < 1e-6), 1);
        if isempty(pair)
            pair = idx;
        end
        alphaS(end + 1) = -real(pole); %#ok<AGROW>
        betaS(end + 1) = -imag(pole); %#ok<AGROW>
        BS(end + 1) = real(residue); %#ok<AGROW>
        CS(end + 1) = imag(residue); %#ok<AGROW>
        used(idx) = true;
        used(pair) = true;
    end
end
end

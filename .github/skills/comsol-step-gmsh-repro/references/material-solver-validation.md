# Fit boundary materials and reproduce the transient solver setup

## Contents

1. Convert COMSOL boundary data
2. Fit EDG reflection coefficients
3. Build the EDG case entrypoint
4. Validate and compare results

## 1. Convert COMSOL boundary data

Let `Z0 = rho0*c0`. Determine the physical quantity stored by every COMSOL table or feature before fitting.

- From normalized specific admittance `Y`:

  `R(omega) = (1 - Z0*Y(omega)) / (1 + Z0*Y(omega))`

- From impedance `Z`:

  `R(omega) = (Z(omega) - Z0) / (Z(omega) + Z0)`

- From absorption `alpha` with no phase model, a real approximation sometimes used by the source model is `R = sqrt(1-alpha)`. Document this loss of phase information.

Confirm the surface-normal/sign convention against the EDG characteristic flux. A correct magnitude with the wrong sign or reciprocal quantity can produce an apparently stable but physically incorrect boundary.

## 2. Fit EDG reflection coefficients

Fit the reflection transfer function, not the raw admittance, to the representation consumed by `AbsorbBC`:

`R(s) = RI + sum_k AS_k/(s+lambda_k) + 1/2 sum_k [(BS_k+i CS_k)/(s+alpha_k+i beta_k) + (BS_k-i CS_k)/(s+alpha_k-i beta_k)]`.

Require `lambda_k > 0`, `alpha_k > 0`, and conjugate pairing. Store MATLAB arrays:

- `RI`;
- `AS`, `lambdaS` for real poles;
- `BS`, `CS`, `alphaS`, `betaS` for complex pairs;
- `freq`, `trueValue`, `ApproxValue`;
- `rms_error`, `max_error`, `max_abs_R`.

Choose the fit band from the source spectrum, mesh resolution, and COMSOL material definition. Check passivity over the simulation band plus a documented margin. Pole count and RMS tolerance are case-specific; increase complexity only when diagnostics show a material error relevant to the excitation band.

Plot at least magnitude and phase or real and imaginary parts. A small global RMS can hide a large local error near the carrier frequency.

## 3. Build the EDG case entrypoint

Use `assets/templates/edg_main.py.template` and make these choices explicit:

- mesh path and physical-label dictionary;
- `rho0`, `c0`, spatial order, time order, CFL, and end time;
- one boundary parameter for every mesh label;
- fitted `.mat` file for each frequency-dependent boundary;
- receiver coordinates and COMSOL output time sequence;
- source kind and normalization.

For a COMSOL `NormalVelocity` source with zero initial pressure/velocity, use the EDG zero initial condition plus prescribed boundary normal velocity. Do not substitute an initial monopole pulse merely because an older example uses one. Match carrier frequency, amplitude, delay, sigma/width, phase, and baseline. A Taylor/ADER integrator may need source time derivatives at each stage; use the solver's supported prescribed-boundary mechanism.

Before the full run, instantiate the simulation and compute the actual `dt`/step count. Abort with a clear message when the mesh makes the requested run impractical.

Save enough metadata to reproduce the result:

- `prec`, `prec_times`, receiver coordinates;
- `dt`, step count, requested/actual total time;
- mesh filename and element/order data;
- `rho0`, `c0`, CFL, source kind/label/parameters;
- boundary parameters or material provenance.

## 4. Validate and compare results

Use three levels of validation:

1. Structural tests: MPH parsing, physical-label coverage, fit coefficient shape, stable poles, passivity, source waveform, and output schema.
2. Smoke run: a few time steps to prove the code path, mesh loading, boundary initialization, finite values, and serialization.
3. Physical run: the full requested time, compared with COMSOL at identical receivers and sample times.

A smoke trace may be all zero when it ends before source onset or propagation to the receiver. State this explicitly; it is not evidence that the physical result is correct.

For aligned traces `p_edg` and `p_ref`, report:

- absolute RMS: `sqrt(mean((p_edg-p_ref)^2))`;
- maximum absolute error: `max(abs(p_edg-p_ref))`;
- relative L2: `norm(p_edg-p_ref)/norm(p_ref)` with zero-reference handling;
- the same metrics over physically meaningful windows such as pre-arrival, direct field, first reflection, and late tail.

Plot COMSOL and EDG together with a separate error panel. Explain whether discrepancies arise from source normalization, mesh dispersion, material fitting, boundary mapping, missing physics, or interpolation. Do not set a universal tolerance: record the case-specific acceptance threshold and why it is appropriate.

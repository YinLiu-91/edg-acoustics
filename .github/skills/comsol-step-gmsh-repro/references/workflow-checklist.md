# End-to-end COMSOL acoustics reproduction checklist

Use this checklist before claiming that a COMSOL case has been reproduced. A recovered mesh is only one intermediate artifact.

## 1. Inputs and provenance

- [ ] Locate the `.mph`, `.step`, imported tables, existing `.geo/.msh`, solver scripts, COMSOL receiver export, and any model documentation.
- [ ] Record the COMSOL creation/last-computation version and the installed COMSOL version.
- [ ] Record geometry units, acoustic domain count, expected physical dimensions, physics interface, study, and solution time range.
- [ ] Preserve original inputs; write generated reports and meshes to new paths.

Useful commands:

```bash
file case.mph
unzip -l case.mph | sed -n '1,120p'
/path/to/comsol version
```

## 2. MPH semantics

- [ ] Extract or parse `dmodel.xml`, `smodel.json`, `modelinfo.xml`, and relevant `resources/*` members.
- [ ] List named `SelectionFeature` boundary groups with tag, name, dimension, entity IDs, and entity count.
- [ ] List active `PhysicsFeature` nodes, including `Impedance`, `NormalVelocity`, `SoundHard`, sources, and explicit selections.
- [ ] Resolve `/selection/<tag>` references and filter negative sentinel IDs.
- [ ] Record evaluated `rho0`, `c0`, impedance/admittance parameters, source function parameters, receiver coordinates, solver order, time range, and output range.
- [ ] Resolve conflicts in favor of active physics and document every override.
- [ ] Extract imported material/function tables from `resources/*` or document their external source.
- [ ] For every active fitted/interpolated material, record the referenced function tag and distinguish its evaluated response from imported/raw source data.

## 3. Physical-label mapping

- [ ] Assign stable integer labels for every EDG boundary type and one volume label.
- [ ] Confirm active non-hard groups are mutually disjoint after precedence resolution.
- [ ] Define default hard wall as all exterior acoustic boundaries minus active special groups.
- [ ] Keep inactive speakers or diagnostic covers separate when that prevents accidental source assignment.
- [ ] Generate a table containing label, name, COMSOL selection/feature, entities/count, equation, and EDG treatment.

## 4. Mesh-path decision

For STEP/OCC/Gmsh:

- [ ] Import with the intended `Geometry.OCCScaling` and `Geometry.OCCImportLabels` settings.
- [ ] Check volume count, bbox, surface count/range, and coverage of all referenced COMSOL entity IDs.
- [ ] Stop if IDs do not map; do not assign by position or numeric coincidence.
- [ ] Generate `Physical Surface` and `Physical Volume` groups only after mapping validation.

For COMSOL mesh export:

- [ ] Identify the component and mesh tag that use the intended geometry or virtual operations.
- [ ] Run that mesh in COMSOL and export linear NASTRAN shell and solid elements with geometry information.
- [ ] Confirm triangle/tetra property/entity references survive the export.
- [ ] Convert with an explicit entity-to-label mapping and fail on unknown or duplicate references.

## 5. Mesh acceptance

- [ ] Run `gmsh -check case.msh -`.
- [ ] Verify expected triangle physical labels are present and non-empty.
- [ ] Verify tetrahedra have the expected acoustic volume label.
- [ ] Verify boundary triangle counts equal the sum of physical-group counts.
- [ ] Verify no missing, duplicate, or unknown entity assignment.
- [ ] Record points, triangles, tetrahedra, bbox, minimum/median triangle area, tetra volume, edge length, and insphere diameter.
- [ ] Inspect generation warnings such as invalid surface elements, PLC errors, or ill-shaped tetrahedra.
- [ ] Compute the actual EDG `dt` using the intended mesh/order/CFL and estimate the full-run step count.
- [ ] Reject or explicitly flag a mesh whose small features make the requested transient run impractical.

## 6. Boundary equations and material fitting

- [ ] Confirm whether source data represent impedance `Z`, admittance `Y`, absorption, or reflection `R`.
- [ ] If active physics references a COMSOL function, export that function through COMSOL as the primary fit target; do not substitute the imported table or `p:fitteddata`.
- [ ] Retain raw/imported material samples as provenance and quantify their mismatch from the fitted active boundary.
- [ ] Convert to EDG's reflection convention using the same `rho0*c0` as the case.
- [ ] Choose and document fit frequency band, sample weighting, pole count, iterations, and error limits.
- [ ] Require stable real/complex poles and verify `max |R| <= 1` over the simulation band plus a documented margin.
- [ ] Save `RI`, real-pole coefficients, complex-pole coefficients, source/fit samples, RMS/max error, and passivity metric in each `.mat`.
- [ ] Save active function/target provenance, rational order, raw-table diagnostic errors, and the passivity frequency limit.
- [ ] Plot magnitude/phase or real/imaginary diagnostics and inspect large local errors even when RMS passes.

## 7. EDG entrypoint

- [ ] Map every mesh physical label to exactly one EDG boundary parameter.
- [ ] Use zero initial state for a boundary-driven COMSOL source unless the COMSOL model actually specifies a nonzero initial field.
- [ ] Reproduce normal-velocity/source waveform, amplitude, frequency, delay, sigma/width, phase, and derivative handling required by the time integrator.
- [ ] Match `rho0`, `c0`, receiver coordinates, spatial order, time order, CFL, end time, and COMSOL output times.
- [ ] Guard against excessive estimated steps before allocating or running the full case.
- [ ] Write output metadata including mesh name, labels/BC parameters, receiver coordinates, `dt`, output times, orders, CFL, and source information.

## 8. Verification and comparison

- [ ] Unit-test MPH parsing, selection precedence, boundary coverage, parameter extraction, source waveform, and `.mat` passivity.
- [ ] Run a short smoke test to verify imports, mesh loading, boundary initialization, finite values, and result serialization.
- [ ] State explicitly that a smoke test before wave arrival does not validate the acoustic result.
- [ ] Run the requested physical duration and compare against COMSOL at identical receiver locations and sample times.
- [ ] Report absolute RMS, maximum absolute error, relative L2 error, and useful time-window errors (for example pre-arrival, direct field, and reflected/reverberant tail).
- [ ] Explain dominant discrepancies: geometry/mesh, boundary fit, source normalization, solver dispersion, sampling, or missing COMSOL physics.

## Completion record

The case README or report must contain:

- input provenance and versions;
- boundary/physical-label table;
- mesh generation and validation commands;
- mesh diagnostics and time-step budget;
- material conversion formulas, fit settings, and errors;
- EDG run command and result schema;
- COMSOL comparison method and metrics;
- known gaps between recovered physical groups and fully reproduced equations.

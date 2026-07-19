# Recover COMSOL model semantics from `.mph`

## Contents

1. Archive inspection
2. Selections and active physics
3. Parameters, functions, sources, and outputs
4. Boundary precedence and reporting
5. Failure modes

## 1. Archive inspection

Treat `.mph` as a ZIP archive before relying on the COMSOL GUI or Java API. Member paths vary by COMSOL release, so enumerate the archive and locate by basename when necessary.

```bash
file case.mph
unzip -l case.mph | sed -n '1,120p'
unzip -p case.mph dmodel.xml > /tmp/case_dmodel.xml
unzip -p case.mph smodel.json > /tmp/case_smodel.json
unzip -p case.mph modelinfo.xml > /tmp/case_modelinfo.xml
```

Common sources:

- `dmodel.xml`: model feature tree, selections, physics features, geometry/mesh history, functions, studies, and datasets.
- `smodel.json`: evaluated parameters and solution metadata.
- `modelinfo.xml`: model title, dimension, interfaces, and version metadata.
- `resources/*`: imported impedance/admittance tables, interpolation data, or auxiliary files.

Record the last-computation version. A newer local COMSOL may usually load an older model, but the archive remains the source of truth for boundary semantics and provenance.

## 2. Selections and active physics

Parse every relevant `SelectionFeature`:

- preserve `tag`, `name`, and `op`;
- read the dimension and explicit entity list;
- filter non-positive values such as `-454,-1`;
- preserve helper selections separately from solver boundary groups.

Parse every active `PhysicsFeature` in the acoustic interface:

- `Impedance`: record model type, constant `Z`/`Y`, rational approximation reference, and interpolation function;
- `NormalVelocity`: record selection and expression such as `vn(t)`;
- `SoundHard`: distinguish default all-boundary behavior from an explicit selection;
- other sources or losses: record their operator, selection, and parameters even when EDG has no implementation yet.

Resolve both forms of selection:

- named references such as `/selection/sel12`;
- feature-local `selection/explicit` entity lists.

An explicit selection with no entity list can mean a default feature applying to all exterior boundaries. Treat it as a default that is overridden by more specific active features.

## 3. Parameters, functions, sources, and outputs

Walk `smodel.json` recursively and retain parameter name, original expression, evaluated real/imaginary value, unit, and description. At minimum recover:

- acoustic constants `rho0` and `c0`;
- impedance/admittance constants and absorption coefficients;
- source amplitude, carrier frequency, delay, width/sigma, phase, and function definition;
- study start/end time and requested output expression such as `range(0,T0/40,Tend)`;
- receiver/probe coordinates and exported pressure component;
- mesh/component/study tags needed for Java automation.

Inspect function features and `resources/*` together. A physics feature may reference an interpolation or rational approximation feature by tag rather than embed coefficients directly.

Do not assume that a function named “impedance” stores `Z`: inspect units, expressions, and the physics feature consuming it to determine whether it represents `Z`, `Y`, absorption, or reflection.

## 4. Boundary precedence and reporting

Build the solver mapping in this order:

1. Collect all exterior acoustic-domain boundary entities.
2. Assign active sources and special physics features.
3. Assign active frequency-dependent and constant impedance/admittance features.
4. Assign explicit hard-wall or inactive diagnostic groups.
5. Define default hard wall as the remaining exterior boundaries.

If a helper selection and active physics conflict, active physics wins. Report the overwritten entity IDs so the decision is reviewable.

Write a raw semantics report before assigning EDG labels. Include:

- model/version metadata;
- all recovered selections;
- all relevant physics features and parameters;
- evaluated scalar parameters;
- resource member names;
- unresolved named references or expressions.

Then write a reviewed mapping table with columns:

| label | name | COMSOL feature | selection | entities/count | COMSOL equation | EDG treatment |
|---:|---|---|---|---:|---|---|

## 5. Failure modes

- If archive members use unexpected paths, search by basename instead of assuming the root path.
- If a named selection cannot be resolved, stop mapping that physics feature and report the missing tag.
- If the same entity receives two active non-default features, resolve from COMSOL feature precedence or request model clarification; never silently choose.
- If parameters remain symbolic, use COMSOL evaluation/export or reconstruct dependencies explicitly before copying numbers.
- If the local COMSOL cannot load the model version, continue archive inspection and STEP validation; do not fabricate missing evaluated settings.

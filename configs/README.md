# Experiment configurations

Keep any checked-in, named experiment configuration here only after the
simulator and experiment runner have stable interfaces. Use the same settings
for every controller in a comparison and record the seed, scenario, and
parameter values in generated result metadata.

Local overrides belong in `.env`, which is ignored by Git. Start with
`.env.example`; do not commit credentials or machine-specific paths.

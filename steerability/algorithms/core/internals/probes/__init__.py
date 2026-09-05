"""Probes: calibrated affine readouts over model internals.

`Probe` is model-free feature math with canonical polarity (a score at or above zero means
present). `fit_probe` and `calibrate_bias` fit and calibrate probes from contrastive pairs;
`ProbeSet` scores many probes in one read-only forward and returns a `ProbeReadings`;
`ProbeSetFit` defers fitting to the model a pipeline provides at steer time.
"""
from .fitting import CalibrationSpec, ProbeEvaluation, ProbeFitSpec, calibrate_bias, evaluate_probe, fit_probe
from .probe import Probe
from .probe_set import ProbeReadings, ProbeSet, ProbeSetFit

__all__ = [
    "CalibrationSpec",
    "Probe",
    "ProbeEvaluation",
    "ProbeFitSpec",
    "ProbeReadings",
    "ProbeSet",
    "ProbeSetFit",
    "calibrate_bias",
    "evaluate_probe",
    "fit_probe",
]

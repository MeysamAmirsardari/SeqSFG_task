"""Two-tone coherence: does onset asynchrony break the binding of two tones, and does it do so
the way the temporal-coherence model says?

A behavioural counterpart to Figure 8B of Elhilali, Ma, Micheyl, Oxenham & Shamma (2009),
measured with the asynchrony-detection task of their Figure 2, generalised so that the onset
asynchrony of the precursor sequence is a parametric variable rather than a two-level contrast.

    tcoh.config       every parameter, and the validator that refuses impossible ones
    tcoh.model        the temporal-coherence model, reduced to two channels: the prediction
    tcoh.stimulus     building and rendering one trial, and the invariants that make it a test
    tcoh.track        the adaptive staircase, and the simulation that says what it measures
    tcoh.psychometric psychometric functions: the simulated listener and the fit
    tcoh.design       the order of the session
    tcoh.observer     simulated listeners: the hypothesis and its rivals
    tcoh.runner       running a session with a person
    tcoh.analysis     thresholds, the coherence index, the tests, the power
    tcoh.verify       everything checkable without a listener
    tcoh.plots        figures
"""
__version__ = "0.1.0"

# CorDA-derived PCA

`CordaPCA` is an activation-steering adaptation of the context-oriented weight
decomposition in [Yang et al., 2024](https://arxiv.org/abs/2406.05223). Its steering
variant follows [steering-lite at 0a064ba](https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/corda_pca.py).

The CorDA paper uses weights and a calibration dataset to initialize trainable
adapters. Here, the decomposition is used to construct a steering vector instead.
For each target Linear module, the fit pools positive and negative input activations
to form a damped, uncentered second-moment matrix. This matrix and the module's
weights define the CorDA basis. Paired positive-minus-negative differences are
expressed in that basis; their first centered principal component is oriented toward
the mean difference and mapped back to an output-space steering vector.

Inference adds `strength * direction` to each token's module output. Model weights
stay fixed, and only the resulting vector is needed at inference; the decomposition
is not retained. Supplying `directions` reuses fitted output vectors without fitting
again. The PCA vector has unit norm before reconstruction; the output vector need
not have unit norm.

<!-- Authored by PI/Astra. -->

Numerically zero centered differences raise instead of selecting an arbitrary PCA
direction. This includes constant pair differences. The separate `pca_pairwise`
estimator elsewhere in steerability is not a `CordaPCA` option.

::: steerability.algorithms.state_control.corda_pca
    handler: python
    options:
        show_if_no_docstring: true
        show_source: true
        show_root_heading: true
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
        show_submodules: true
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        filters:
          - "!.*Args$"
          - "!^registry"
          - "!^STEERING_METHOD"

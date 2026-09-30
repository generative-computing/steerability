# S-space

`SSpace` implements a weight-SVD activation-steering variant. The method draws
on [S-Space Steering for Eval-Awareness Control in Reasoning Models](https://apartresearch.com/project/sspace-steering-for-evalawareness-control-in-reasoning-models-7j1i)
by Michael J Clark; this control follows [steering-lite at 0a064ba](https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/sspace.py).

For each target Linear module, the fit decomposes its weight matrix as
`W = U diag(s) Vᵀ`. It expresses positive and negative module outputs in coordinates
`z = (output - bias) U / sqrt(s)` and takes their mean difference. `rank` retains the
coordinates with the largest absolute contrast, not necessarily the largest singular
values. These are weight-scaled coordinates; their activation covariance is not
necessarily identity.

At inference, the retained basis is used to read each token's coordinates and map
the edit back to the module output. `cosine` scales the edit by the absolute cosine
with the fitted direction; `off` applies a constant edit. `signed` keeps the cosine's
sign: with positive strength it reinforces either pole of the axis, rather than
always pushing toward the positive examples. Reversing a direction leaves `signed`
unchanged, but reverses the `cosine` and `off` edits.

Unlike CorDA-PCA's fixed output vector, the gated variants need the retained basis
at inference. Model weights stay fixed. For fp16/bf16 outputs, application uses
float32 arithmetic with autocast disabled, then restores the output dtype.

<!-- Authored by PI/Astra. -->

::: steerability.algorithms.state_control.sspace
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

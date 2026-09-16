# Linear-AcT

`LinearAcT` adapts coordinate-wise affine activation transport from
[Rodriguez et al., ICLR 2025](https://openreview.net/forum?id=l2zFn6TIQi), following
[steering-lite at 0a064ba](https://github.com/wassname/steering-lite/blob/0a064ba0c23a4998637ff41c5ab0fb5ca50a4271/src/steering_lite/variants/linear_act.py).
The fit treats negative activations as the source and positive activations as the
target. It sorts samples independently within each coordinate and fits a scalar
slope and bias by least squares. Equal sample counts are required, but pairing is
not used after sorting. This is not a standard-deviation-ratio map.

Inference applies `h + strength * (slope * h + bias - h)` at each selected decoder
layer's output, for every token. Only the slope and bias vectors are needed; model
weights stay fixed. `affine` accepts an already fitted map. This port implements the
coordinate-wise map, without support masking or sequential layerwise fitting.

<!-- Authored by PI/Astra. -->

::: steerability.algorithms.state_control.linear_act
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

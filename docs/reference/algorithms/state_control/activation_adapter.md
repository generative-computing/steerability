# ActivationAdapter

The `require_coverage` argument (default `True`) sets how `steer()` treats a behavior layer that the bound transform has no direction for. With `True`, `steer()` raises `ValueError`. With `False`, the layer is hooked and its hidden states pass through unchanged. An intervention control that freezes to an `activation_adapter` entry records the setting of its bound intervention in that entry, e.g., a `CAST` control whose behavior vector covers a subset of `behavior_layer_ids` freezes with `require_coverage=False`.

::: steerability.algorithms.state_control.activation_adapter
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

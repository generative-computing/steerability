# PRewrite

On chat input, `adapt_messages` places the chosen rewrite before the content of the leading system message, separated by a blank line, and a chat without a leading system message receives it as its system message. A caller that needs the instruction to replace an existing system message composes `SystemPromptFormatter(mode="replace")` directly.

::: steerability.algorithms.input_control.prewrite
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
        show_root_heading: true
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        filters:
          - "!^_"
          - "!.*Args$"
          - "!^registry"
          - "!^STEERING_METHOD"

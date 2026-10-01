# CPO

On chat input, `adapt_messages` places the chosen prompt of each query before the content of the leading system message, separated by a blank line, and a chat without a leading system message receives it as its system message. A caller that needs the instruction to replace an existing system message composes `SystemPromptFormatter(mode="replace")` directly. The proposer is `prompt_lm` with its `prompt_tokenizer` when `prompt_lm` is set, and the pipeline's model with the pipeline tokenizer otherwise.

::: steerability.algorithms.input_control.cpo
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

# Concepts

Steerability structures steering methods, termed controls in the toolkit, into four categories: input, structural, state,
and output. To learn more about these categories, and what determines the category of a control, please see the
conceptual [guide on controls](controls.md).

The toolkit additionally allows controls from different categories to be composed into a single operation on the
model. These composed controls are referred to as steering pipelines. For a conceptual outline of what steering
pipelines are, please see the [guide on pipelines](steering_pipelines.md).

Alongside steering, the toolkit reads model internals for detection through calibrated probes, which drive
conditional steering and routed decoding. For the conceptual overview of detection, please see the
[guide on probes](probes.md).

Steered pipelines can also be written down as portable `.spipe` bundles that contain both the configuration and the
products of an expensive steer step. For the conceptual overview of saving, sharing, and loading pipelines, please
see the [guide on sharing pipelines](spipe.md).

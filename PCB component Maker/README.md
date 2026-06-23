# PCB Component Maker

This folder is for standalone scripts that generate PCB component geometry and later apply component-specific defects.

Current focus:

- `smd_resistor_ceramic_core.py`
  Generates a simple ceramic core STL for an SMD resistor body.
- `female_pin_connector_1x1.py`
  Opens a small UI and live `vedo` preview for a square plastic tube built from outer-minus-inner square blocks, then boolean-subtracted by the connector pin enlarged by `0.001 mm`, keeping only the largest remaining plastic body and using the connector pin centroid as the origin.
- `female_pin_array_builder.py`
  Opens a small UI and live `vedo` preview for a pick-driven center mate: choose 2 parallel triangles on the connector pin, then 2 on the plastic covering, constrain the moving part along the resulting plane normal, and save the mate JSON.

Notes:

- The generator uses a normalized shape, so you can rescale the STL later if needed.
- The goal is reusable component geometry first, then defect logic on top of it.

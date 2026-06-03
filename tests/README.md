# tests/

Unit tests. Critical correctness gates (per §11 of the plan):

- **Causal-validity test** — verifies the teacher's synchronous diagnostic response
  is removed from the D-formatted prefix during latency replay.
- **Payload-equivalence (SHA-256)** — asserts byte-identical normalized diagnostic
  payloads across B/C/C′/D for matched (prefix, edit) triggers.
- **Partial-file probe** — pyrefly snapshot mid-edit on deliberately broken files;
  bounds diagnostic volume and validates the parse-validity gate.

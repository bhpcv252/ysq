# Documentation

Longer notes and derivations. Interface-level documentation lives with each
module, in `src/<Module>/README.md`.

| Document                          | Contents                                             |
| --------------------------------- | ---------------------------------------------------- |
| [architecture.md](architecture.md) | Layering, dependency rules, backend selection        |
| [math.md](math.md)                 | Conventions, derivations and coefficient tables for `Math` |
| [units.md](units.md)               | Conversion factors, their sources, and which are exact |
| `physics.md`                       | Models, approximations and their validity (not yet written) |
| [rendering.md](rendering.md)       | Rendering conventions, why RayTracer is a fragment shader, scene upload, shader embedding |

The unwritten documents land with the modules they describe. Read the relevant
one before working on a module.

---
name: scene-editor
description: 'Interactive scene editing and manipulation. Use when: modifying scenes in the editor, adding/removing objects, adjusting materials interactively, or debugging scene construction.'
---

# Scene Editor

The `scene_editor` executable — interactive scene manipulation with real-time preview.

## Source Map

The scene editor shares `src/scene/` with `ppt_analyze` and `photon_tracer`.
Editor-specific code is in `tools/scene_editor/`.

## Usage

```
scene_editor <scene.pbrt>
```

Interactive GLFW window with ImGui controls for camera, materials, and object placement.

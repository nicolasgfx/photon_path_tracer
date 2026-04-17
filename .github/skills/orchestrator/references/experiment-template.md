# Experiment Template — Phase N

Copy this file to `references/phaseN-name.md` and fill in the sections.

## Phase N — [Name]

[One-sentence description of what this phase adds to the pipeline.]

## Skills Required (in order)

1. **skill-name** — [brief role in this phase]
2. ...

## Config Overrides

```cpp
// Overrides vs the previous phase or defaults
config.param = value;  // reason
```

## Execution Steps

### Step N — [Stage Name] ([skill-name] skill)

**Goal**: [What this step accomplishes]

**Entry point**: [File + function]

**Call sequence**:
```
function_a()
function_b()
```

**Files**:
- `src/path/file.h` — [role]

**Outputs**:
- [What this step produces for downstream consumption]

**Gate check**: [How to verify this step succeeded before proceeding]

**Known gotchas**:
- [Edge cases, common failures, parameter sensitivities]

---

*(Repeat Step section for each stage)*

## End-to-End Verification

1. [Launch command]
2. [Expected visual/numeric result]
3. [Interaction test]
4. [Stability check]

## Transition to Phase N+1

[What the next phase adds and what changes are needed.]

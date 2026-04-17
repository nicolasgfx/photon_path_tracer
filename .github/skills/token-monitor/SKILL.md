---
name: token-monitor
description: 'Token usage and skill cost monitoring. Use when: asking which skill was used, how many tokens a request cost, what the active skills were, token breakdown per turn, session token totals, skill cost estimate, cost of recent request, which SKILL.md files were loaded.'
---

# Token Monitor

Reports LLM token usage and skill activity by parsing the Copilot Chat debug logs for the current session.

## When to Use
- "Which skill was used for my last request?"
- "How many tokens did that cost?"
- "Show me the token usage for this session"
- "What's the cost of the active skills?"

## Key Concepts

**Token counts (exact):** Taken from `llm_request` events in `main.jsonl`. Each model round-trip records `inputTokens` and `outputTokens`.

**Skill activity (exact):** Taken from `discovery` events (name: `"Load Skills"`) in `main.jsonl`. These log which `SKILL.md` files were resolved for each turn.

**Per-skill token estimate (approximate):** Each active SKILL.md contributes ~`file_size / 4` input tokens to the context window. This is a rough approximation — the real input token count also includes conversation history, system prompt, open files, and all other context.

---

## Procedure

### Step 1 — Find the current session log

Run in terminal:
```powershell
# Find the most recently written main.jsonl across all sessions
$log = Get-ChildItem "$env:APPDATA\Code\User\workspaceStorage\*\GitHub.copilot-chat\debug-logs\*\main.jsonl" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
Write-Host "Log: $log"
Write-Host "Size: $([math]::Round((Get-Item $log).Length / 1KB, 1)) KB"
```

Store the path in `$log` for all subsequent commands.

### Step 2 — Extract LLM token usage per turn

Stream the log with Node.js (safe for large files):
```powershell
node -e "
const fs = require('fs');
const path = '$log'.replace(/\\/g, '/');
const lines = fs.readFileSync(path, 'utf8').split('\n').filter(Boolean);
const reqs = lines.map(l => { try { return JSON.parse(l); } catch { return null; } })
  .filter(e => e && e.type === 'llm_request');
let totalIn = 0, totalOut = 0;
console.log('Turn | Model | Input Tokens | Output Tokens | TTFT (ms) | Duration (ms)');
console.log('-----|-------|--------------|---------------|-----------|---------------');
reqs.forEach((r, i) => {
  const a = r.attrs || {};
  totalIn += a.inputTokens || 0;
  totalOut += a.outputTokens || 0;
  console.log((i+1) + ' | ' + (a.model||'?') + ' | ' + (a.inputTokens||0) + ' | ' + (a.outputTokens||0) + ' | ' + (a.ttft||0) + ' | ' + (r.dur||0));
});
console.log('');
console.log('SESSION TOTAL: ' + totalIn + ' input, ' + totalOut + ' output (' + (totalIn+totalOut) + ' total)');
"
```

### Step 3 — Extract skill activity per turn

```powershell
node -e "
const fs = require('fs');
const path = '$log'.replace(/\\/g, '/');
const lines = fs.readFileSync(path, 'utf8').split('\n').filter(Boolean);
const events = lines.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
const skillEvents = events.filter(e => e.type === 'discovery' && e.name === 'Load Skills');
if (!skillEvents.length) { console.log('No skill discovery events found.'); process.exit(0); }
skillEvents.forEach((e, i) => {
  const d = (e.attrs||{}).details||'';
  const match = d.match(/loaded: \[([^\]]*)\]/);
  const skills = match ? match[1] : '(none)';
  console.log('Turn ~' + (i+1) + ': ' + skills);
});
"
```

### Step 4 — Estimate per-skill token overhead

For each skill name found in Step 3, read its SKILL.md size from the workspace:
```powershell
$skillsRoot = "d:\Dev\photon_path_tracer\photon_path_tracer\.github\skills"
Get-ChildItem "$skillsRoot\*\SKILL.md" | ForEach-Object {
    $name = $_.Directory.Name
    $bytes = $_.Length
    $est = [math]::Round($bytes / 4)
    Write-Host "$name : $bytes bytes (~$est input tokens)"
}
```

**Note:** The `agent-customization` and `troubleshoot` skills are copilot built-ins and live outside the workspace. Their SKILL.md files are not readable via the workspace path.

### Step 5 — Present the report

Format the output as:

```
## Token Usage Report — Current Session

### Per-Turn LLM Costs
| Turn | Model | Input | Output | Total | TTFT |
|------|-------|-------|--------|-------|------|
|  1   | claude-sonnet-4.x | 18 432 | 312 | 18 744 | 2 100 ms |
...

### Session Totals
- Total input tokens: X
- Total output tokens: Y
- Grand total: Z
- Turns: N

### Active Skills This Session (Estimated Input Token Cost)
| Skill | SKILL.md size | Est. tokens |
|-------|--------------|-------------|
| renderer | 8 400 B | ~2 100 |
| code-quality | 3 200 B | ~800 |
...

### Notes
- Token counts are exact from llm_request log events.
- Per-skill estimates = SKILL.md bytes ÷ 4 (rough; actual input includes full conversation context).
- Subagent calls appear in child log files and are not included in parent turn totals.
```

---

## Limitations

| Limitation | Detail |
|-----------|--------|
| Per-skill attribution is approximate | inputTokens = skill + system prompt + history + all open files |
| Built-in skills not measurable | `agent-customization`, `troubleshoot` are copilot-internal; no file path available |
| Subagent tokens not aggregated | Child sessions (`runSubagent-*.jsonl`) have separate token counts not summed here |
| Log file must exist | If no log is found, the session was started in a mode without debug logging |

## Log Format Reference

```
$env:APPDATA\Code\User\workspaceStorage\<wsId>\GitHub.copilot-chat\debug-logs\<sessionId>\main.jsonl
```

Relevant event types:

| Type | Key attrs | Notes |
|------|-----------|-------|
| `llm_request` | `model`, `inputTokens`, `outputTokens`, `ttft` | One per model round-trip |
| `discovery` (name: `Load Skills`) | `details` (has `loaded: [skill1, skill2]`) | One per turn where skills were checked |
| `user_message` | `content` | Marks the start of each user turn |

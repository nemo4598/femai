# MCP Server Configuration for KayenAI

This file documents the recommended MCP (Model Context Protocol) servers for working efficiently with KayenAI.

## Enabled Servers

### GitHub MCP
**Purpose:** Collaborate on issues, PRs, code search, and project tracking

**Use cases:**
- Search codebase across components (find where RoPE is implemented)
- Create issues for component milestones
- Track PRs for code review
- Reference commits in development notes

**Configuration:**
- Requires GitHub personal access token (PAT)
- Scope: `repo`, `read:org` (adjust as needed)

### Filesystem MCP
**Purpose:** Enhanced file operations for large datasets, checkpoints, and model weights

**Use cases:**
- Browse checkpoint directories and dataset splits
- Manage large binary files safely
- Track file modifications across training runs
- Version dataset preprocessing outputs

## To Set Up

1. **GitHub:** Provide PAT when connecting MCP
2. **Filesystem:** Grant access to KayenAI project root

---

Note: Additional MCPs (Bash, Database, Memory) can be added later if development patterns require them.

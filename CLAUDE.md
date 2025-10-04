# Claude Code Preferences

## Error Handling
- **Avoid large try/catch blocks** - Let errors happen naturally
- Only use try/catch for specific, known exceptions with <3 lines of handling code
- Prefer explicit error checking over broad exception handling
- Let stack traces provide debugging information

## Code Style
- Keep functions focused and single-purpose
- Prefer clarity over cleverness
- Use type hints where helpful
- Avoid over-engineering simple tasks

## Testing Commands
- Run lint and typecheck after code changes:
  ```bash
  # Add project-specific lint/typecheck commands here
  # e.g., pytest, flake8, mypy
  ```

## Project-Specific Notes
- Using SQLite for caching stock data
- Cache provider handles insert/update automatically
- Prefer simple, direct implementations
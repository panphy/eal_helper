# AGENTS.md - AI Assistant Guide for EAL Helper

This document provides essential context for AI assistants working on the EAL Learning Companion codebase.

## Project Overview

**EAL Helper** is a Streamlit web application that supports English as Additional Language (EAL) learners reading challenging academic content. Students paste passages and receive:
- Simplified, level-appropriate English versions
- Full translations (13+ languages)
- Vocabulary scaffolding with bilingual support
- Comprehension check questions

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11 |
| Frontend Framework | Streamlit |
| AI Processing | OpenAI API (GPT-5-mini) |
| Styling | Custom CSS embedded in Python |
| Dev Environment | Docker devcontainer |

## Project Structure

```
eal_helper/
├── .devcontainer/
│   └── devcontainer.json    # VS Code/Codespaces dev environment
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── app.py                   # Main application (single-file architecture)
├── requirements.txt         # Python dependencies
├── README.md                # User-facing documentation
└── AGENTS.md                # This file
```

## Key File: app.py

The application uses a single-file architecture. Here's how it's organized:

| Lines | Section | Purpose |
|-------|---------|---------|
| 1-22 | Configuration | Imports, language mappings, CEFR levels, limits |
| 23-180 | Helper Functions | Type-safe getters, language detection, preferences |
| 182-595 | CSS Design System | Custom CSS with design tokens and utility classes |
| 600-833 | AI Core | OpenAI integration with JSON schema validation |
| 856-1162 | UI Components | Streamlit layout, inputs, outputs, progress states |

## Development Commands

```bash
# Start the development server
streamlit run app.py

# Start with CORS/XSRF disabled (for local dev)
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

The server runs on **port 8501** with hot-reload enabled.

## Environment Variables

| Variable | Description | Location |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Required for AI processing | Streamlit secrets (`st.secrets`) |

## Code Conventions

### Type Hints
Always use type hints for function signatures:
```python
def parse_protected_terms(raw: str) -> list[str]:
def get_scaffolded_content(text: str, language: str, cefr_level: str, protected: list[str]) -> dict | None:
```

### State Management
- Initialize session state with null checks and defaults
- Use `on_change` callbacks to reset results when inputs change
- Persist user preferences to `.eal_helper_prefs.json`

### Error Handling
- Custom exceptions: `TranslationConstraintError`, `ProtectedTermError`
- Safe type extraction with `safe_get_str()` and `safe_get_list()`
- JSON schema validation for AI outputs

### CSS Organization
CSS is embedded in `app.py` using `st.html()`. Follow the design system:
- **Colors**: Use CSS variables (e.g., `--color-primary: #2f6bff`)
- **Utilities**: `.stack`, `.row`, `.card`, `.pill`, `.box`
- **States**: `.success-box`, `.info-alert`, `.warning-state`

### Code Comments
- Use section headers with `# -----` separators
- Document complex prompts and validation logic
- Keep inline comments minimal but meaningful

## AI Integration Details

### OpenAI Configuration
- Model: `gpt-5.4-mini`
- Response format: Structured JSON with schema validation
- Retry logic: 3 attempts with varied system prompts

### Output Schema
The AI returns JSON with this structure:
```json
{
  "simplified_text": "...",
  "full_translation": "...",
  "vocabulary": [
    {"word": "...", "english_definition": "...", "translated_definition": "..."}
  ],
  "comprehension_questions": ["...", "...", "..."]
}
```

### Constraints
- Protected terms must appear verbatim in simplified text
- Full translations must not contain English (validated with stopword heuristics)
- Vocabulary limited to 5 words per request

## User Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Session quota | 20 API calls | Prevent abuse |
| Rate limit | 3 calls / 60 seconds | Prevent rapid-fire requests |
| Input length | 4,000 characters | API cost control |

## Supported Languages

Arabic, Chinese (Simplified), Chinese (Traditional), French, German, Japanese, Polish, Portuguese, Russian, Spanish, Thai, Turkish, Urdu

## CEFR Levels

- **A2 (Beginner)**: Simple vocabulary, short sentences
- **B1 (Intermediate)**: Moderate complexity, common academic terms
- **B2 (Advanced)**: More complex structures, fuller academic vocabulary

## Git Workflow

### Branch Naming
- Feature branches: `panphy/codex/*` or `panphy/Codex/*`
- Development follows PR-based workflow with code review

### Commit Style
- Clear, descriptive commit messages
- Focus on single concerns per commit
- Recent history shows UI/UX polish and incremental improvements

## Testing Guidelines

Currently no automated tests. When making changes:
1. Run the app locally with `streamlit run app.py`
2. Test all user flows: simplification, translation, vocabulary, comprehension
3. Verify rate limiting and quota behavior
4. Check mobile responsiveness

## Common Tasks

### Adding a New Language
1. Add to `LANGUAGES` dict in app.py (around line 10)
2. Format: `"Language Name": "language_code"`

### Modifying the AI Prompt
1. Locate `get_scaffolded_content()` function (line ~600)
2. Update `SYSTEM_PROMPTS` list for robust fallback behavior
3. Ensure JSON schema in `response_format` matches expected output

### Updating the Theme
1. Edit `.streamlit/config.toml` for Streamlit-level theming
2. Edit CSS in app.py (lines 182-595) for custom component styling

### Changing Rate Limits
1. Modify constants near top of app.py:
   - `MAX_CALLS_PER_SESSION = 20`
   - `RATE_LIMIT_CALLS = 3`
   - `RATE_LIMIT_WINDOW = 60`

## External Services

| Service | Purpose | Notes |
|---------|---------|-------|
| OpenAI API | Text processing | Requires API key in secrets |
| PanPhy CDN | Logo and favicon | URLs hardcoded in app.py |

## Accessibility Considerations

- Proper color contrast ratios maintained
- Semantic HTML structure
- Clear visual feedback for all states
- Character counter with color-coded warnings

## Performance Optimizations

- `@st.cache_resource` for OpenAI client reuse
- `ThreadPoolExecutor` for non-blocking AI calls
- Progress animation simulated client-side during processing

## Known Limitations

- Single-file architecture may become unwieldy as features grow
- No automated testing suite
- User preferences stored locally (not portable across sessions)
- AI outputs require heuristic validation (not guaranteed correct)

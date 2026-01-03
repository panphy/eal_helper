Academic Text Helper is a lightweight web app built to support EAL learners when reading challenging subject content. Students paste in a paragraph from a lesson, worksheet, or textbook, and the app produces a clearer, level-appropriate version of the same ideas without losing key meaning. It also provides a full translation of the original text, plus a focused vocabulary scaffold to help students learn the academic language rather than just “look up” answers.

What it does
	•	Simplifies academic English to a chosen CEFR level (A2, B1, B2)
	•	Preserves key terms using an optional “protected vocabulary” list (useful for science and other technical subjects)
	•	Translates the full original text into the student’s chosen language
	•	Builds a vocabulary table with:
	•	the target word
	•	a simple English definition
	•	a translation of the word
	•	a translation of the definition
	•	Generates quick comprehension questions to check understanding

Why this exists

EAL students often understand the concept but get blocked by dense vocabulary, long sentences, or unfamiliar academic phrasing. This tool is designed to reduce that language barrier while encouraging progress in English, making it easier for students to engage with the same curriculum content as their peers.

Notes

This app uses an OpenAI model to generate outputs. As with any AI tool, it can occasionally simplify too much or miss nuance, so it should be used as a scaffold, not as a substitute for teacher explanations or mark schemes.

AI usage limits

To keep usage fair and predictable, the app enforces a per-session quota and rate limiting for AI requests. Each browser session is limited to a fixed number of AI calls; once the quota is reached, the app will ask the user to start a new session. Requests are also throttled to prevent rapid, repeated calls in a short time window.

Tech stack

	•	Backend: Python + Flask
	•	AI: OpenAI API
	•	Frontend: HTML, CSS, JavaScript (served by Flask templates)

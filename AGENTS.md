# Project Rules

## TRACE manuscript source

- The authoritative TRACE manuscript is
  `paper/AuthorKit27/AuthorKit27/paper_trace_aaai27.tex`.
- Unless the user explicitly names another manuscript file, make TRACE paper
  edits only in this authoritative source and do not modify legacy copies.
- Do not search, inspect, or analyze implementation code when reviewing or
  editing the manuscript unless the user explicitly asks for code inspection
  or states that code should be used.  Base manuscript edits only on the paper
  and other sources explicitly placed in scope by the user.

## Manuscript experiment prose

- Do not include experiment-progress statements in the manuscript, such as
  saying that an experiment is unfinished, ongoing, pending, or will be added
  later.
- Write experimental sections in final-paper form, under the working
  assumption that the planned evidence is complete and supports the stated,
  bounded conclusions.
- When an exact value is not yet available in the workspace, retain a `TBD`
  placeholder without discussing progress or inventing a result.

## Manuscript terminology and writing style

- In every TRACE manuscript edit, use established, widely understood terms
  from information retrieval, machine learning, and deep learning whenever
  possible.
- Do not invent terminology or use rare words to make the method sound more
  novel.  If a common technical term expresses the same meaning, use the common
  term.
- Avoid over-packaging through marketing language, stacked adjectives,
  unnecessary component names, or inflated claims.  Describe the problem,
  mechanism, and evidence directly.
- Keep terminology consistent across the abstract, main text, equations,
  figures, tables, and appendix.  Introduce a method-specific term only when it
  is necessary, precisely defined, and used consistently afterward.
- Prefer clear, concise sentences over ornate wording.  A reviewer should be
  able to understand each claim without interpreting uncommon vocabulary or
  newly coined phrases.

## Figure generation

- **Never overwrite existing figure files.** When generating a modified or new
  version of a figure, always save it with a suffix appended to the original
  filename (e.g., `figure2_adjusted.pdf` instead of `figure2.pdf`, or
  `figure3_v2.pdf` instead of `figure3.pdf`).  The original file must remain
  untouched so it can always be reverted to.

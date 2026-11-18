// One entry per backend status (app/answer/service.py). `tone` drives the badge colour;
// `sourcesLabel` says honestly what the listed clauses are.

export const STATUS = {
  answered: {
    label: 'Answered · citations verified',
    tone: 'ok',
    sourcesLabel: 'Sources cited',
  },
  insufficient_sources: {
    label: 'The sources do not answer this',
    tone: 'warn',
    sourcesLabel: 'Closest clauses found',
  },
  not_found: {
    label: 'Not in coverage',
    tone: 'muted',
    sourcesLabel: null,
  },
  no_coverage: {
    label: 'No rules captured for this date',
    tone: 'muted',
    sourcesLabel: null,
  },
  ungrounded: {
    label: 'Unverified · the answer cited no source',
    tone: 'bad',
    sourcesLabel: 'Clauses retrieved (not cited)',
  },
  llm_unavailable: {
    label: 'Answer unavailable · relevant clauses below',
    tone: 'warn',
    sourcesLabel: 'Most relevant clauses',
  },
}

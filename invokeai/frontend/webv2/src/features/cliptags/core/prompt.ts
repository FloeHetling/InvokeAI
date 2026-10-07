export type ClipTagHotPrefix = '~' | '*';

export const CLIP_TAG_HOT_PREFIXES: readonly ClipTagHotPrefix[] = ['~', '*'];

export interface ClipTagPromptQuery {
  /** The text typed after the hot prefix, exactly as in the prompt. */
  query: string;
  /** From the hot prefix to the caret; selecting a tag replaces it. */
  range: { end: number; start: number };
}

/** Matches the backend, which treats underscores and spaces alike. */
export const normalizeClipTagQuery = (query: string): string => query.replaceAll('_', ' ').replace(/\s+/g, ' ').trim();

export const CLIP_TAG_MIN_QUERY_LENGTH = 2;

export const isClipTagQueryEligible = (query: string): boolean =>
  normalizeClipTagQuery(query).length >= CLIP_TAG_MIN_QUERY_LENGTH;

/** Longest search text looked for behind the caret, so a long line costs a bounded scan. */
const MAX_QUERY_LENGTH = 100;

const isWordCharacter = (character: string | undefined): boolean =>
  character !== undefined && /[\p{L}\p{N}\p{M}_]/u.test(character);

/**
 * The hot-prefix search the caret is inside, if any. The prefix has to start a word, so `~` inside text stays
 * literal. The search text may contain spaces (tags do) and ends at a comma or the end of the line.
 */
export const getActiveClipTagQuery = (
  value: string,
  caret: number,
  hotPrefix: ClipTagHotPrefix
): ClipTagPromptQuery | null => {
  const limit = Math.max(0, caret - MAX_QUERY_LENGTH - 1);

  for (let index = caret - 1; index >= limit; index--) {
    const character = value[index];

    if (character === '\n' || character === ',') {
      return null;
    }

    if (character === hotPrefix) {
      return isWordCharacter(value[index - 1])
        ? null
        : { query: value.slice(index + 1, caret), range: { end: caret, start: index } };
    }
  }

  return null;
};

/**
 * What to write for a chosen tag: the rendered text and a separator, taking over a comma already sitting after the
 * caret so selecting in the middle of a list does not double it.
 */
export const getClipTagInsertion = (
  value: string,
  range: { end: number; start: number },
  renderedContent: string
): { range: { end: number; start: number }; text: string } => {
  const following = /^[ \t]*,[ \t]*/.exec(value.slice(range.end));

  return {
    range: { end: range.end + (following?.[0].length ?? 0), start: range.start },
    text: `${renderedContent}, `,
  };
};

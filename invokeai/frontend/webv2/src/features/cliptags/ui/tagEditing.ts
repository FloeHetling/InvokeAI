import type { ClipTagDetail, ClipTagType, ClipTagUpdate } from '@features/cliptags/core/types';

export interface TagFormValues {
  content: string;
  /** Digits only; empty means the popularity is unknown. */
  popularity: string;
  tagSetIds: ReadonlySet<string>;
  type: ClipTagType;
}

export const getTagFormValues = (tag: ClipTagDetail): TagFormValues => ({
  content: tag.content,
  popularity: tag.popularity === null ? '' : String(tag.popularity),
  tagSetIds: new Set(tag.tagSetIds),
  type: tag.type,
});

/** Null when the text is not a whole number the server accepts; an empty field is a valid "unknown". */
export const parsePopularity = (text: string): { value: number | null } | null => {
  const trimmed = text.trim();

  if (trimmed === '') {
    return { value: null };
  }

  const value = /^\d+$/.test(trimmed) ? Number(trimmed) : Number.NaN;

  return Number.isSafeInteger(value) ? { value } : null;
};

const sameMembers = (left: ReadonlySet<string>, right: readonly string[]): boolean =>
  left.size === right.length && right.every((id) => left.has(id));

/** Only what changed, so a stale form cannot overwrite membership or fields it never touched; null when nothing did. */
export const buildTagUpdate = (tag: ClipTagDetail, values: TagFormValues): ClipTagUpdate | null => {
  const update: ClipTagUpdate = {};
  const content = values.content.trim();
  const popularity = parsePopularity(values.popularity);

  if (content !== tag.content) {
    update.content = content;
  }

  if (values.type !== tag.type) {
    update.type = values.type;
  }

  if (popularity !== null && popularity.value !== tag.popularity) {
    update.popularity = popularity.value;
  }

  if (!sameMembers(values.tagSetIds, tag.tagSetIds)) {
    update.tagSetIds = [...values.tagSetIds];
  }

  return Object.keys(update).length > 0 ? update : null;
};

import type { ClipTagBulkSelection, ClipTagFilter } from '@features/cliptags/core/types';

/** The backend refuses a bulk request that names more ids than this; "all matching" is the way past it. */
export const MAX_BULK_IDS = 1000;

/**
 * Tags the user marked for a bulk operation. The server resolves `matching` from the filter, so rows that were
 * never loaded can be part of it. It keeps the filter the user selected under, which is exactly what is sent, and
 * the server's total only to label the selection.
 */
export type TagSelection =
  | { ids: ReadonlySet<string>; mode: 'ids' }
  | { filter: ClipTagFilter; mode: 'matching'; total: number };

export const NO_TAG_SELECTION: TagSelection = { ids: new Set<string>(), mode: 'ids' };

export const isSameTagFilter = (a: ClipTagFilter, b: ClipTagFilter): boolean =>
  a.search === b.search &&
  a.tagSetId === b.tagSetId &&
  a.tagType === b.tagType &&
  (a.uncategorized ?? false) === (b.uncategorized ?? false);

export const getSelectedCount = (selection: TagSelection): number =>
  selection.mode === 'matching' ? selection.total : selection.ids.size;

export const isTagSelected = (selection: TagSelection, id: string): boolean =>
  selection.mode === 'matching' || selection.ids.has(id);

/** The rows the header checkbox can select by id: as many loaded rows as one request can name. */
export const getSelectableIds = (loadedIds: readonly string[]): readonly string[] =>
  loadedIds.length > MAX_BULK_IDS ? loadedIds.slice(0, MAX_BULK_IDS) : loadedIds;

/** An id selection that cannot take another row. */
export const isIdSelectionFull = (selection: TagSelection): boolean =>
  selection.mode === 'ids' && selection.ids.size >= MAX_BULK_IDS;

/**
 * Selection is explicit by id, so leaving "all matching" keeps the rows on screen minus the one toggled: the server
 * cannot exclude rows from a filter. A full id selection ignores further additions.
 */
export const toggleTagSelection = (selection: TagSelection, id: string, loadedIds: readonly string[]): TagSelection => {
  const ids = new Set(selection.mode === 'matching' ? getSelectableIds(loadedIds) : selection.ids);

  if (ids.has(id)) {
    ids.delete(id);
  } else if (ids.size < MAX_BULK_IDS) {
    ids.add(id);
  }

  return { ids, mode: 'ids' };
};

export const getSelectAllState = (selection: TagSelection, loadedIds: readonly string[]): boolean | 'indeterminate' => {
  if (selection.mode === 'matching') {
    return true;
  }

  if (selection.ids.size === 0 || loadedIds.length === 0) {
    return false;
  }

  return getSelectableIds(loadedIds).every((id) => selection.ids.has(id)) ? true : 'indeterminate';
};

/** The header checkbox clears a complete selection and otherwise selects the loaded rows, up to the id limit. */
export const toggleLoadedSelection = (selection: TagSelection, loadedIds: readonly string[]): TagSelection =>
  getSelectAllState(selection, loadedIds) === true
    ? NO_TAG_SELECTION
    : { ids: new Set(getSelectableIds(loadedIds)), mode: 'ids' };

export const selectAllMatching = (total: number, filter: ClipTagFilter): TagSelection => ({
  filter,
  mode: 'matching',
  total,
});

/** Rows can vanish under a selection (a merge, a delete elsewhere); only loaded rows stay selected. */
export const pruneTagSelection = (selection: TagSelection, loadedIds: readonly string[]): TagSelection => {
  if (selection.mode === 'matching' || selection.ids.size === 0) {
    return selection;
  }

  const loaded = new Set(loadedIds);
  const ids = [...selection.ids].filter((id) => loaded.has(id));

  return ids.length === selection.ids.size ? selection : { ids: new Set(ids), mode: 'ids' };
};

export const toBulkSelection = (selection: TagSelection): ClipTagBulkSelection =>
  selection.mode === 'matching'
    ? { filter: selection.filter, mode: 'filter' }
    : { ids: [...selection.ids], mode: 'ids' };

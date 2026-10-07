import type { ClipTag } from '@features/cliptags/core/types';

import { vi } from 'vitest';

/**
 * The transport module of the manager, replaced in browser tests: `vi.mock('@features/cliptags/data/api', () =>
 * import('./clipTagApi.testing'))`. Every export is a mock the test drives and inspects; `getClipModelConfig` and
 * `searchClipTags` exist only because the shared query module imports them, and the manager never calls them.
 */
export const bulkMutateClipTags = vi.fn();
export const cancelClipImport = vi.fn();
export const commitClipImport = vi.fn();
export const createClipSyntaxProfile = vi.fn();
export const createClipTagSet = vi.fn();
export const deleteClipSyntaxProfile = vi.fn();
export const deleteClipTag = vi.fn();
export const deleteClipTagSet = vi.fn();
export const downloadClipSampleCsv = vi.fn();
export const getClipModelConfig = vi.fn();
export const getClipTag = vi.fn();
export const getClipTagStatus = vi.fn();
export const listClipSyntaxProfiles = vi.fn();
export const listClipTags = vi.fn();
export const listClipTagSets = vi.fn();
export const prepareClipImport = vi.fn();
export const renameClipTagSet = vi.fn();
export const searchClipTags = vi.fn();
export const stageClipImport = vi.fn();
export const updateClipSyntaxProfile = vi.fn();
export const updateClipTag = vi.fn();

export const TAGS: ClipTag[] = [
  { content: 'red hair', id: 't1', popularity: 1200, type: 'general' },
  { content: 'blue eyes', id: 't2', popularity: null, type: 'general' },
  { content: 'by artist', id: 't3', popularity: 30, type: 'artist' },
];
export const TOTAL = 120;

const allMocks = [
  bulkMutateClipTags,
  cancelClipImport,
  commitClipImport,
  createClipSyntaxProfile,
  createClipTagSet,
  deleteClipSyntaxProfile,
  deleteClipTag,
  deleteClipTagSet,
  downloadClipSampleCsv,
  getClipModelConfig,
  getClipTag,
  getClipTagStatus,
  listClipSyntaxProfiles,
  listClipTags,
  listClipTagSets,
  prepareClipImport,
  renameClipTagSet,
  searchClipTags,
  stageClipImport,
  updateClipSyntaxProfile,
  updateClipTag,
];

/** Every mock back to a healthy backend: available, three tags of a hundred and twenty, one tag set, no profiles. */
export const resetClipTagApi = (): void => {
  for (const mock of allMocks) {
    mock.mockReset();
  }

  getClipTagStatus.mockResolvedValue({ available: true, reason: null });
  listClipTags.mockResolvedValue({ items: TAGS, nextCursor: 'next', total: TOTAL });
  listClipTagSets.mockResolvedValue([{ id: 's1', modelCount: 1, name: 'Anime', tagCount: 3 }]);
  listClipSyntaxProfiles.mockResolvedValue([]);
  getClipTag.mockImplementation((id: string) =>
    Promise.resolve({ ...TAGS.find((tag) => tag.id === id)!, tagSetIds: ['s1'] })
  );
  updateClipTag.mockResolvedValue({ merged: false, tag: { ...TAGS[0]!, tagSetIds: ['s1'] } });
  deleteClipTag.mockResolvedValue(undefined);
  bulkMutateClipTags.mockResolvedValue({ affectedCount: TOTAL, mergedCount: 0, selectedCount: TOTAL });
  cancelClipImport.mockResolvedValue(undefined);
};

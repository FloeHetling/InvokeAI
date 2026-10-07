import type { ClipTagFilter } from '@features/cliptags/core/types';

/** The shared query client is cleared on account change, so keys carry no account part. */
export const clipTagKeys = {
  all: ['cliptags'] as const,
  status: () => ['cliptags', 'status'] as const,
  searches: () => ['cliptags', 'search'] as const,
  search: (query: string, modelId: string | null, category: string | null) =>
    ['cliptags', 'search', { category, modelId, query }] as const,
  /** Every filter's tag list. */
  tagLists: () => ['cliptags', 'tags'] as const,
  tags: (filter: ClipTagFilter) => ['cliptags', 'tags', filter] as const,
  tag: (id: string) => ['cliptags', 'tag', id] as const,
  tagSets: () => ['cliptags', 'tag-sets'] as const,
  syntaxProfiles: () => ['cliptags', 'syntax-profiles'] as const,
  modelConfig: (modelId: string) => ['cliptags', 'model-config', modelId] as const,
};

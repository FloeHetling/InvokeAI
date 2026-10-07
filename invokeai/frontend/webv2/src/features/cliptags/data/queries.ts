import type { ClipTagCandidate, ClipTagCategory, ClipTagFilter, ClipTagPage } from '@features/cliptags/core/types';
import type { InfiniteData, QueryClient } from '@tanstack/react-query';

import { infiniteQueryOptions, queryOptions } from '@tanstack/react-query';

import {
  getClipModelConfig,
  getClipTag,
  getClipTagStatus,
  listClipSyntaxProfiles,
  listClipTags,
  listClipTagSets,
  searchClipTags,
} from './api';
import { clipTagKeys } from './keys';

/** One category page shown to the user; one more is requested so a following page is known to exist. */
export const CLIP_TAG_SEARCH_PAGE_SIZE = 50;

export const CLIP_TAG_LIST_PAGE_SIZE = 50;

const STATUS_STALE_MS = 5 * 60_000;

export const clipTagStatusOptions = () =>
  queryOptions({
    queryFn: ({ signal }) => getClipTagStatus(signal),
    queryKey: clipTagKeys.status(),
    staleTime: STATUS_STALE_MS,
  });

interface SearchPage {
  candidates: ClipTagCandidate[];
  hasMore: boolean;
}

/**
 * Prompt search. Without a category the backend returns its best matches in one response; with one, pages
 * are requested by offset. Superseded requests are aborted by the query client.
 */
export const clipTagSearchOptions = ({
  category,
  modelId,
  query,
}: {
  category: ClipTagCategory | null;
  modelId: string | null;
  query: string;
}) =>
  infiniteQueryOptions({
    getNextPageParam: (lastPage: SearchPage, _pages: SearchPage[], lastOffset: number) =>
      category !== null && lastPage.hasMore ? lastOffset + CLIP_TAG_SEARCH_PAGE_SIZE : undefined,
    initialPageParam: 0,
    queryFn: async ({ pageParam, signal }): Promise<SearchPage> => {
      if (category === null) {
        return { candidates: await searchClipTags({ modelId, query }, signal), hasMore: false };
      }

      const candidates = await searchClipTags(
        { category, limit: CLIP_TAG_SEARCH_PAGE_SIZE + 1, modelId, offset: pageParam, query },
        signal
      );

      return {
        candidates: candidates.slice(0, CLIP_TAG_SEARCH_PAGE_SIZE),
        hasMore: candidates.length > CLIP_TAG_SEARCH_PAGE_SIZE,
      };
    },
    queryKey: clipTagKeys.search(query, modelId, category),
    staleTime: 30_000,
  });

export const clipTagsInfiniteOptions = (filter: ClipTagFilter) =>
  infiniteQueryOptions({
    getNextPageParam: (lastPage: ClipTagPage) => lastPage.nextCursor ?? undefined,
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      listClipTags(filter, { cursor: pageParam, limit: CLIP_TAG_LIST_PAGE_SIZE }, signal),
    queryKey: clipTagKeys.tags(filter),
  });

export const clipTagOptions = (id: string) =>
  queryOptions({ queryFn: ({ signal }) => getClipTag(id, signal), queryKey: clipTagKeys.tag(id) });

export const clipTagSetsOptions = () =>
  queryOptions({ queryFn: ({ signal }) => listClipTagSets(signal), queryKey: clipTagKeys.tagSets() });

export const clipSyntaxProfilesOptions = () =>
  queryOptions({ queryFn: ({ signal }) => listClipSyntaxProfiles(signal), queryKey: clipTagKeys.syntaxProfiles() });

export const clipModelConfigOptions = (modelId: string) =>
  queryOptions({
    queryFn: ({ signal }) => getClipModelConfig(modelId, signal),
    queryKey: clipTagKeys.modelConfig(modelId),
  });

/**
 * Everything the manager shows is derived from the tag database, so any edit refreshes all of it. A refetch walks
 * every loaded page of a tag list one after another, which with thousands of rows loaded keeps the write pending
 * for seconds. Lists are cut back to their first page first (what the user sees is replaced by it, and "Load more"
 * continues from the refreshed page), so a refresh costs one request per list however far the user had scrolled.
 */
export const invalidateClipTagData = (client: QueryClient): Promise<void> => {
  client.setQueriesData<InfiniteData<ClipTagPage, string | null>>({ queryKey: clipTagKeys.tagLists() }, (data) =>
    data && data.pages.length > 1 ? { pageParams: data.pageParams.slice(0, 1), pages: data.pages.slice(0, 1) } : data
  );

  return client.invalidateQueries({ queryKey: clipTagKeys.all });
};

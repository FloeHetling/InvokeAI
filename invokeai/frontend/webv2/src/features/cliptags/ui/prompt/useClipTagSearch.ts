import type { ClipTagCandidate, ClipTagCategory } from '@features/cliptags/core/types';

import { isClipTagQueryEligible, normalizeClipTagQuery } from '@features/cliptags/core/prompt';
import { clipTagSearchOptions, clipTagStatusOptions } from '@features/cliptags/data/queries';
import { keepPreviousData, useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

const NO_CANDIDATES: ClipTagCandidate[] = [];

/** Whether the tag database is usable. Only asked while `enabled`, and shared by every prompt field. */
export const useClipTagStatus = (enabled = true) => useQuery({ ...clipTagStatusOptions(), enabled });

export interface ClipTagSearch {
  candidates: ClipTagCandidate[];
  hasMore: boolean;
  isFetchingMore: boolean;
  /** The text is long enough to search; shorter text gets a hint instead of results. */
  isEligible: boolean;
  isError: boolean;
  isSearching: boolean;
  loadMore: () => void;
}

/**
 * Candidates for a prompt search. The caller debounces `query`; a newer query supersedes an older request, so
 * a late response can never replace the current results.
 */
export const useClipTagSearch = ({
  category,
  enabled,
  modelId,
  query,
}: {
  category: ClipTagCategory | null;
  enabled: boolean;
  modelId: string | null;
  query: string;
}): ClipTagSearch => {
  const normalizedQuery = normalizeClipTagQuery(query);
  const isEligible = enabled && isClipTagQueryEligible(normalizedQuery);
  const search = useInfiniteQuery({
    ...clipTagSearchOptions({ category, modelId, query: normalizedQuery }),
    enabled: isEligible,
    // The previous text's results stay up while the next are fetched, so the list does not flash empty per keystroke.
    placeholderData: keepPreviousData,
  });
  const candidates = useMemo(() => {
    const seen = new Set<string>();
    const unique: ClipTagCandidate[] = [];

    for (const page of search.data?.pages ?? []) {
      for (const candidate of page.candidates) {
        if (!seen.has(candidate.id)) {
          seen.add(candidate.id);
          unique.push(candidate);
        }
      }
    }

    return unique;
  }, [search.data]);
  const { fetchNextPage, hasNextPage, isFetchingNextPage } = search;
  const loadMore = () => {
    if (hasNextPage && !isFetchingNextPage) {
      void fetchNextPage();
    }
  };

  return {
    candidates: isEligible ? candidates : NO_CANDIDATES,
    hasMore: isEligible && hasNextPage,
    isEligible,
    isError: isEligible && search.isError,
    isFetchingMore: isEligible && isFetchingNextPage,
    isSearching: isEligible && search.isPending,
    loadMore,
  };
};

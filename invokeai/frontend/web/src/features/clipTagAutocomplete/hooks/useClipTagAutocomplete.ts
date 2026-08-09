import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  type CtaAutocompleteArgs,
  type CtaAutocompleteFilter,
  useGetCtaStatusQuery,
  useLazyAutocompleteCtaQuery,
} from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';

type CtaAutocompleteCandidate = S['CtaAutocompleteCandidate'];

type UseClipTagAutocompleteInput = {
  query: string;
  modelId: string | null;
  filter: CtaAutocompleteFilter | null;
};

type UseClipTagAutocompleteOutput = {
  candidates: CtaAutocompleteCandidate[];
  isSearching: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  loadMore: () => void;
  normalizedQuery: string;
  isAvailable: boolean;
  isStatusLoading: boolean;
};

type CtaRequestHandle<T> = {
  abort: () => void;
  unsubscribe?: () => void;
  unwrap: () => Promise<T>;
};
type CtaTrigger<T> = (args: CtaAutocompleteArgs) => CtaRequestHandle<T>;
type CtaRequestCallbacks<T> = {
  onStart: () => void;
  onSuccess: (data: T) => void;
  onFailure: () => void;
};

type CtaRequestState = {
  key: string | null;
  candidates: CtaAutocompleteCandidate[];
  isSearching: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
};

const DEBOUNCE_MS = 200;
const FILTERED_PAGE_SIZE = 50;
const FILTERED_REQUEST_LIMIT = FILTERED_PAGE_SIZE + 1;

/** Shared immutable fallback so ineligible/stale queries do not allocate a new result array every render. */
const EMPTY_CANDIDATES: CtaAutocompleteCandidate[] = [];

export const normalizeCtaQuery = (query: string): string => query.replaceAll('_', ' ').replace(/\s+/g, ' ').trim();

export const isCtaQueryEligible = (query: string, available: boolean): boolean =>
  available && normalizeCtaQuery(query).length >= 2;

export function getCtaFilteredPage<T>(data: T[]): { items: T[]; hasMore: boolean } {
  return {
    items: data.slice(0, FILTERED_PAGE_SIZE),
    hasMore: data.length > FILTERED_PAGE_SIZE,
  };
}

/** Owns debounce, abort and stale-result suppression for CTA remote search and filtered pagination. */
export class CtaRequestCoordinator<T> {
  private timer: ReturnType<typeof setTimeout> | undefined;
  private request: CtaRequestHandle<T> | null = null;
  private generation = 0;
  private key: string | null = null;

  schedule(key: string, args: CtaAutocompleteArgs, trigger: CtaTrigger<T>, callbacks: CtaRequestCallbacks<T>): boolean {
    if (this.key === key) {
      return false;
    }

    this.cancel();
    this.key = key;
    const generation = this.generation;

    this.timer = setTimeout(() => {
      this.timer = undefined;
      if (generation !== this.generation || this.key !== key) {
        return;
      }

      callbacks.onStart();
      this.startRequest(generation, key, args, trigger, callbacks);
    }, DEBOUNCE_MS);

    return true;
  }

  requestMore(
    key: string,
    args: CtaAutocompleteArgs,
    trigger: CtaTrigger<T>,
    callbacks: CtaRequestCallbacks<T>
  ): boolean {
    if (this.key !== key || this.timer !== undefined || this.request !== null) {
      return false;
    }

    callbacks.onStart();
    this.startRequest(this.generation, key, args, trigger, callbacks);
    return true;
  }

  cancel(): void {
    this.generation += 1;
    this.key = null;
    clearTimeout(this.timer);
    this.timer = undefined;
    this.request?.abort();
    this.request?.unsubscribe?.();
    this.request = null;
  }

  private startRequest(
    generation: number,
    key: string,
    args: CtaAutocompleteArgs,
    trigger: CtaTrigger<T>,
    callbacks: CtaRequestCallbacks<T>
  ): void {
    const request = trigger(args);
    this.request = request;

    void (async () => {
      try {
        const data = await request.unwrap();
        if (generation === this.generation && this.key === key) {
          callbacks.onSuccess(data);
        }
      } catch {
        if (generation === this.generation && this.key === key) {
          callbacks.onFailure();
        }
      } finally {
        request.unsubscribe?.();
        if (this.request === request) {
          this.request = null;
        }
      }
    })();
  }
}

export function useClipTagAutocomplete({
  query,
  modelId,
  filter,
}: UseClipTagAutocompleteInput): UseClipTagAutocompleteOutput {
  const normalizedQuery = useMemo(() => normalizeCtaQuery(query), [query]);

  const { data: status, isLoading: isStatusLoading } = useGetCtaStatusQuery(undefined, {
    refetchOnFocus: false,
    refetchOnReconnect: false,
  });
  const [trigger] = useLazyAutocompleteCtaQuery();
  const [coordinator] = useState(() => new CtaRequestCoordinator<CtaAutocompleteCandidate[]>());
  const [requestState, setRequestState] = useState<CtaRequestState>({
    key: null,
    candidates: EMPTY_CANDIDATES,
    isSearching: false,
    isLoadingMore: false,
    hasMore: false,
  });
  const isAvailable = status?.available === true;
  const requestKey = isCtaQueryEligible(normalizedQuery, isAvailable)
    ? `${normalizedQuery}:${modelId ?? ''}:${filter ?? ''}`
    : null;

  useEffect(() => {
    if (requestKey === null) {
      coordinator.cancel();
      return;
    }

    coordinator.schedule(
      requestKey,
      {
        q: normalizedQuery,
        model_id: modelId ?? undefined,
        tag_filter: filter ?? undefined,
        offset: filter === null ? undefined : 0,
        limit: filter === null ? undefined : FILTERED_REQUEST_LIMIT,
      },
      trigger,
      {
        onStart: () => {
          setRequestState({
            key: requestKey,
            candidates: EMPTY_CANDIDATES,
            isSearching: true,
            isLoadingMore: false,
            hasMore: false,
          });
        },
        onSuccess: (data) => {
          if (filter === null) {
            setRequestState({
              key: requestKey,
              candidates: data,
              isSearching: false,
              isLoadingMore: false,
              hasMore: false,
            });
            return;
          }
          const page = getCtaFilteredPage(data);
          setRequestState({
            key: requestKey,
            candidates: page.items,
            isSearching: false,
            isLoadingMore: false,
            hasMore: page.hasMore,
          });
        },
        onFailure: () => {
          setRequestState({
            key: requestKey,
            candidates: EMPTY_CANDIDATES,
            isSearching: false,
            isLoadingMore: false,
            hasMore: false,
          });
        },
      }
    );
  }, [coordinator, filter, modelId, normalizedQuery, requestKey, trigger]);

  useEffect(() => () => coordinator.cancel(), [coordinator]);

  const isCurrentRequest = requestKey !== null && requestState.key === requestKey;
  const loadMore = useCallback(() => {
    if (
      requestKey === null ||
      filter === null ||
      !isCurrentRequest ||
      requestState.isSearching ||
      requestState.isLoadingMore ||
      !requestState.hasMore
    ) {
      return;
    }

    coordinator.requestMore(
      requestKey,
      {
        q: normalizedQuery,
        model_id: modelId ?? undefined,
        tag_filter: filter,
        offset: requestState.candidates.length,
        limit: FILTERED_REQUEST_LIMIT,
      },
      trigger,
      {
        onStart: () => {
          setRequestState((state) => (state.key === requestKey ? { ...state, isLoadingMore: true } : state));
        },
        onSuccess: (data) => {
          const page = getCtaFilteredPage(data);
          setRequestState((state) => {
            if (state.key !== requestKey) {
              return state;
            }
            return {
              ...state,
              candidates: [...state.candidates, ...page.items],
              isLoadingMore: false,
              hasMore: page.hasMore,
            };
          });
        },
        onFailure: () => {
          setRequestState((state) =>
            state.key === requestKey ? { ...state, isLoadingMore: false, hasMore: false } : state
          );
        },
      }
    );
  }, [
    coordinator,
    filter,
    isCurrentRequest,
    modelId,
    normalizedQuery,
    requestKey,
    requestState.candidates.length,
    requestState.hasMore,
    requestState.isLoadingMore,
    requestState.isSearching,
    trigger,
  ]);

  return {
    candidates: isCurrentRequest ? requestState.candidates : EMPTY_CANDIDATES,
    isSearching: isCurrentRequest ? requestState.isSearching : false,
    isLoadingMore: isCurrentRequest ? requestState.isLoadingMore : false,
    hasMore: isCurrentRequest ? requestState.hasMore : false,
    loadMore,
    normalizedQuery,
    isAvailable,
    isStatusLoading,
  };
}

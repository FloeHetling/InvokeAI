import type { ClipTagCandidate, ClipTagCategory, ClipTagPromptQuery } from '@features/cliptags/contracts';
import type { GenerateLora, GenerateModelConfig } from '@features/generation/core/types';
import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type { PromptTriggerKey, PromptTriggerQuery } from '@features/generation/ui/promptFields/promptFocus';
import type { PromptTriggerOption } from '@features/generation/ui/promptFields/promptTriggerOptions';
import type { CompositionEvent, KeyboardEvent, ReactNode } from 'react';

import {
  CLIP_TAG_CATEGORIES,
  getActiveClipTagQuery,
  getClipTagInsertion,
  isClipTagQueryEligible,
  normalizeClipTagQuery,
} from '@features/cliptags/contracts';
import { useClipTagSearch, useClipTagStatus } from '@features/cliptags/react';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { getTextareaCaretRect } from '@features/generation/ui/promptFields/promptCaret';
import { getActiveTriggerQuery, insertPromptText } from '@features/generation/ui/promptFields/promptFocus';
import { PromptTagAutocomplete } from '@features/generation/ui/promptFields/PromptTagAutocomplete';
import { PromptTriggerAutocomplete } from '@features/generation/ui/promptFields/PromptTriggerAutocomplete';
import {
  getInlineTriggerOptions,
  usePromptTriggerOptions,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import { DismissOnViewportChange } from '@features/generation/ui/promptFields/useDismissOnViewportChange';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useCallback, useId, useMemo, useRef, useState } from 'react';

const CARET_KEYS = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
const LIST_KEYS = ['ArrowUp', 'ArrowDown'];

/** A trigger completes from the model's own vocabulary; a tag search asks the tag database. */
type AutocompleteState =
  | { caretRect: CaretRect; kind: 'trigger'; query: PromptTriggerQuery; textarea: HTMLTextAreaElement }
  | { caretRect: CaretRect; kind: 'tag'; query: ClipTagPromptQuery; textarea: HTMLTextAreaElement };

/** Tag searches wait for typing to pause; the list follows the settled text. */
const TAG_SEARCH_DEBOUNCE_MS = 200;

export interface PromptTriggerAutocompleteApi {
  comboboxProps: {
    'aria-activedescendant': string | undefined;
    'aria-autocomplete': 'list';
    'aria-controls': string | undefined;
    'aria-expanded': boolean;
    role: 'combobox';
    onCompositionEnd: (event: CompositionEvent<HTMLTextAreaElement>) => void;
    onCompositionStart: () => void;
    onKeyUp: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  };
  element: ReactNode;
  isOpen: boolean;
  close: () => void;
  refresh: (textarea: HTMLTextAreaElement | null) => void;
  handleKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

const TAG_CATEGORY_CYCLE: readonly (ClipTagCategory | null)[] = [null, ...CLIP_TAG_CATEGORIES];

export const usePromptTriggerAutocomplete = ({
  isDisabled = false,
  keys,
  loras,
  onChange,
  selectedModel,
}: {
  keys: readonly PromptTriggerKey[];
  loras: GenerateLora[];
  selectedModel: GenerateModelConfig | undefined;
  isDisabled?: boolean;
  onChange: (value: string) => void;
}): PromptTriggerAutocompleteApi => {
  const listboxId = useId();
  const optionIdPrefix = `${listboxId}-option-`;
  const [state, setState] = useState<AutocompleteState | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [category, setCategory] = useState<ClipTagCategory | null>(null);
  const [settledTagQuery, setSettledTagQuery] = useState('');
  const isComposingRef = useRef(false);
  const tagQueryTimerRef = useRef<number | null>(null);
  const options = usePromptTriggerOptions(loras, selectedModel);
  const { clipTags } = useGenerationUi();
  const { data: tagStatus } = useClipTagStatus(clipTags.enabled);
  const isTagSearchAvailable = clipTags.enabled && tagStatus?.available === true;
  const isTagSearchOpen = state?.kind === 'tag';
  const liveTagQuery = state?.kind === 'tag' ? normalizeClipTagQuery(state.query.query) : '';
  const isTagQuerySettled = liveTagQuery === settledTagQuery;
  const tagSearch = useClipTagSearch({
    category,
    enabled: isTagSearchOpen,
    modelId: selectedModel?.key ?? null,
    query: settledTagQuery,
  });
  const tagCandidates = tagSearch.candidates;
  const isTagSearchPending = !isTagQuerySettled || tagSearch.isSearching;

  const matches = useMemo(
    () => (state?.kind === 'trigger' ? getInlineTriggerOptions(options, state.query.key, state.query.query) : []),
    [options, state]
  );
  const itemCount = state?.kind === 'tag' ? tagCandidates.length : matches.length;
  const currentIndex = Math.min(activeIndex, Math.max(0, itemCount - 1));
  // A tag search stays open while it has nothing to show, so it can say why.
  const isOpen = state !== null && (state.kind === 'tag' || matches.length > 0);

  const clearTagQueryTimer = useCallback(() => {
    if (tagQueryTimerRef.current !== null) {
      window.clearTimeout(tagQueryTimerRef.current);
      tagQueryTimerRef.current = null;
    }
  }, []);
  const close = useCallback(() => {
    clearTagQueryTimer();
    // The next search starts from its own text, not from what was typed last time.
    setSettledTagQuery('');
    setState(null);
  }, [clearTagQueryTimer]);

  useMountEffect(() => clearTagQueryTimer);

  const scheduleTagQuery = useCallback(
    (query: string) => {
      clearTagQueryTimer();
      tagQueryTimerRef.current = window.setTimeout(() => {
        tagQueryTimerRef.current = null;
        setSettledTagQuery(query);
      }, TAG_SEARCH_DEBOUNCE_MS);
    },
    [clearTagQueryTimer]
  );

  const refresh = useCallback(
    (textarea: HTMLTextAreaElement | null) => {
      if (!textarea || isDisabled || isComposingRef.current) {
        close();
        return;
      }

      setActiveIndex(0);

      const tagQuery =
        isTagSearchAvailable && textarea.selectionStart === textarea.selectionEnd
          ? getActiveClipTagQuery(textarea.value, textarea.selectionStart, clipTags.hotPrefix)
          : null;
      const triggerQuery = getActiveTriggerQuery(textarea.value, textarea.selectionStart, keys);
      // The one that begins closest to the caret is what is being typed: `~red <emb` completes the embedding.
      const isTagSearch =
        tagQuery !== null && (triggerQuery === null || tagQuery.range.start > triggerQuery.range.start);

      if (isTagSearch) {
        const caretRect = getTextareaCaretRect(textarea, tagQuery.range.start);

        if (caretRect) {
          setState({ caretRect, kind: 'tag', query: tagQuery, textarea });
          scheduleTagQuery(normalizeClipTagQuery(tagQuery.query));
          return;
        }
      }

      clearTagQueryTimer();

      const caretRect = triggerQuery ? getTextareaCaretRect(textarea, triggerQuery.range.start) : null;

      setState(triggerQuery && caretRect ? { caretRect, kind: 'trigger', query: triggerQuery, textarea } : null);
    },
    [clearTagQueryTimer, clipTags.hotPrefix, close, isDisabled, isTagSearchAvailable, keys, scheduleTagQuery]
  );

  const selectOption = useCallback(
    (option: PromptTriggerOption) => {
      if (state?.kind === 'trigger') {
        insertPromptText({
          onChange,
          range: state.query.range,
          text: option.value,
          textarea: state.textarea,
          value: state.textarea.value,
        });
      }

      close();
    },
    [close, onChange, state]
  );

  const selectTag = useCallback(
    (candidate: ClipTagCandidate) => {
      if (state?.kind === 'tag') {
        const insertion = getClipTagInsertion(state.textarea.value, state.query.range, candidate.renderedContent);

        insertPromptText({
          onChange,
          range: insertion.range,
          text: insertion.text,
          textarea: state.textarea,
          value: state.textarea.value,
        });
      }

      close();
    },
    [close, onChange, state]
  );

  const changeCategory = useCallback((next: ClipTagCategory | null) => {
    setCategory(next);
    setActiveIndex(0);
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>): void => {
      if (!isOpen || event.nativeEvent.isComposing || event.keyCode === 229) {
        return;
      }

      if (event.altKey) {
        // Alt+Arrow steps through the tag categories, which the buttons offer to the pointer only.
        if (state?.kind === 'tag' && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
          const step = event.key === 'ArrowRight' ? 1 : -1;
          const index = TAG_CATEGORY_CYCLE.indexOf(category);

          event.preventDefault();
          changeCategory(
            TAG_CATEGORY_CYCLE[(index + step + TAG_CATEGORY_CYCLE.length) % TAG_CATEGORY_CYCLE.length] ?? null
          );
        }

        return;
      }

      if (CARET_KEYS.includes(event.key)) {
        return;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        close();
        return;
      }

      if (itemCount === 0) {
        // While the search is still looking, Enter and Tab are meant for its answer: a newline now would land in the
        // middle of the search text.
        if (
          state?.kind === 'tag' &&
          (event.key === 'Enter' || event.key === 'Tab') &&
          isTagSearchPending &&
          isClipTagQueryEligible(liveTagQuery)
        ) {
          event.preventDefault();
        }

        return;
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        const step = event.key === 'ArrowDown' ? 1 : -1;

        event.preventDefault();
        setActiveIndex((current) => (Math.min(current, itemCount - 1) + step + itemCount) % itemCount);
        return;
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        if (state?.kind === 'tag') {
          const candidate = tagCandidates[currentIndex];

          if (candidate) {
            event.preventDefault();
            selectTag(candidate);
          }

          return;
        }

        const option = matches[currentIndex];

        if (option) {
          event.preventDefault();
          selectOption(option);
        }
      }
    },
    [
      category,
      changeCategory,
      close,
      currentIndex,
      isOpen,
      isTagSearchPending,
      itemCount,
      liveTagQuery,
      matches,
      selectOption,
      selectTag,
      state?.kind,
      tagCandidates,
    ]
  );

  const handleKeyUp = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      // Up and Down move the caret between lines when there is no list to move through.
      if (CARET_KEYS.includes(event.key) || (itemCount === 0 && LIST_KEYS.includes(event.key))) {
        refresh(event.currentTarget);
      }
    },
    [itemCount, refresh]
  );

  const handleCompositionStart = useCallback(() => {
    isComposingRef.current = true;
    close();
  }, [close]);

  const handleCompositionEnd = useCallback(
    (event: CompositionEvent<HTMLTextAreaElement>) => {
      isComposingRef.current = false;
      refresh(event.currentTarget);
    },
    [refresh]
  );

  return {
    close,
    comboboxProps: {
      'aria-activedescendant': isOpen && itemCount > 0 ? `${optionIdPrefix}${currentIndex}` : undefined,
      'aria-autocomplete': 'list',
      'aria-controls': isOpen && itemCount > 0 ? listboxId : undefined,
      'aria-expanded': isOpen,
      onCompositionEnd: handleCompositionEnd,
      onCompositionStart: handleCompositionStart,
      onKeyUp: handleKeyUp,
      role: 'combobox',
    },
    element:
      isOpen && state ? (
        <>
          <DismissOnViewportChange dismiss={close} enabled />
          {state.kind === 'tag' ? (
            <PromptTagAutocomplete
              activeIndex={currentIndex}
              candidates={tagCandidates}
              caretRect={state.caretRect}
              category={category}
              hasMore={tagSearch.hasMore}
              isError={tagSearch.isError}
              isFetchingMore={tagSearch.isFetchingMore}
              isSearching={isTagSearchPending}
              isShortQuery={!isClipTagQueryEligible(liveTagQuery)}
              listboxId={listboxId}
              optionIdPrefix={optionIdPrefix}
              onCategoryChange={changeCategory}
              onLoadMore={tagSearch.loadMore}
              onSelect={selectTag}
            />
          ) : (
            <PromptTriggerAutocomplete
              activeIndex={currentIndex}
              caretRect={state.caretRect}
              listboxId={listboxId}
              optionIdPrefix={optionIdPrefix}
              options={matches}
              onSelect={selectOption}
            />
          )}
        </>
      ) : null,
    handleKeyDown,
    isOpen,
    refresh,
  };
};

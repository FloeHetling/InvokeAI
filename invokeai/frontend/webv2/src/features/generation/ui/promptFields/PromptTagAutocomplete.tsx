import type { ClipTagCandidate, ClipTagCategory } from '@features/cliptags/contracts';
import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type { MouseEvent, UIEvent } from 'react';

import { Box, chakra, HStack, Portal, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { CLIP_TAG_CATEGORIES, CLIP_TAG_TYPE_PALETTE } from '@features/cliptags/contracts';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const LIST_WIDTH_PX = 320;
const MAX_HEIGHT_PX = 280;
const CARET_GAP_PX = 4;
const VIEWPORT_MARGIN_PX = 8;
const LOAD_MORE_THRESHOLD_PX = 48;
const OPTION_HOVER_CSS = { bg: 'bg.hover' };
const POPULARITY_FORMAT = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, notation: 'compact' });

const CATEGORY_LABEL_KEYS: Record<ClipTagCategory, string> = {
  artist: 'cliptags.prompt.filterArtist',
  character: 'cliptags.prompt.filterCharacter',
  other: 'cliptags.prompt.filterOther',
};

export interface PromptTagAutocompleteProps {
  activeIndex: number;
  candidates: readonly ClipTagCandidate[];
  caretRect: CaretRect;
  category: ClipTagCategory | null;
  hasMore: boolean;
  isError: boolean;
  isFetchingMore: boolean;
  /** The text is too short to search yet. */
  isShortQuery: boolean;
  isSearching: boolean;
  listboxId: string;
  optionIdPrefix: string;
  onCategoryChange: (category: ClipTagCategory | null) => void;
  onLoadMore: () => void;
  onSelect: (candidate: ClipTagCandidate) => void;
}

/** Tag suggestions at the caret. Focus stays in the prompt: every pointer interaction here is a mousedown that is prevented. */
export const PromptTagAutocomplete = ({
  activeIndex,
  candidates,
  caretRect,
  category,
  hasMore,
  isError,
  isFetchingMore,
  isSearching,
  isShortQuery,
  listboxId,
  onCategoryChange,
  onLoadMore,
  onSelect,
  optionIdPrefix,
}: PromptTagAutocompleteProps) => {
  const { t } = useTranslation();
  const listRef = useRef<HTMLDivElement | null>(null);
  const spaceBelow = window.innerHeight - (caretRect.y + caretRect.height);
  const height = Math.max(
    0,
    Math.min(MAX_HEIGHT_PX, Math.max(spaceBelow, caretRect.y) - VIEWPORT_MARGIN_PX - CARET_GAP_PX)
  );
  const opensBelow = spaceBelow >= height + CARET_GAP_PX + VIEWPORT_MARGIN_PX;
  const left = Math.max(
    VIEWPORT_MARGIN_PX,
    Math.min(caretRect.x, window.innerWidth - LIST_WIDTH_PX - VIEWPORT_MARGIN_PX)
  );
  // Opening upward anchors the bottom edge to the caret, so a short list stays next to it.
  const position = opensBelow
    ? { top: `${caretRect.y + caretRect.height + CARET_GAP_PX}px` }
    : { bottom: `${window.innerHeight - caretRect.y + CARET_GAP_PX}px` };
  // Results of the previous text stay on screen while the next ones load, so the status only speaks for an empty list.
  const status = isShortQuery
    ? t('cliptags.prompt.typeMore')
    : isError
      ? t('cliptags.prompt.searchFailed')
      : candidates.length > 0
        ? null
        : isSearching
          ? t('cliptags.prompt.searching')
          : t('cliptags.prompt.noMatches');

  useLayoutEffect(() => {
    listRef.current
      ?.querySelector(`#${CSS.escape(`${optionIdPrefix}${activeIndex}`)}`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, optionIdPrefix]);

  const handleScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const list = event.currentTarget;

      if (hasMore && list.scrollHeight - list.scrollTop - list.clientHeight < LOAD_MORE_THRESHOLD_PX) {
        onLoadMore();
      }
    },
    [hasMore, onLoadMore]
  );

  return (
    <Portal>
      <Stack
        aria-label={t('cliptags.prompt.search')}
        bg="bg.muted"
        borderColor="border.emphasized"
        borderRadius="md"
        borderWidth="1px"
        boxShadow="md"
        data-prompt-autocomplete=""
        gap="0"
        left={`${left}px`}
        maxH={`${height}px`}
        position="fixed"
        role="group"
        w={`${LIST_WIDTH_PX}px`}
        zIndex="popover"
        {...position}
      >
        <HStack
          aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight"
          aria-label={t('cliptags.prompt.filters')}
          flexShrink={0}
          gap="1"
          px="2"
          py="1"
          role="group"
          wrap="wrap"
        >
          <CategoryChip
            isActive={category === null}
            label={t('cliptags.prompt.filterAll')}
            value={null}
            onSelect={onCategoryChange}
          />
          {CLIP_TAG_CATEGORIES.map((value) => (
            <CategoryChip
              key={value}
              isActive={category === value}
              label={t(CATEGORY_LABEL_KEYS[value])}
              value={value}
              onSelect={onCategoryChange}
            />
          ))}
        </HStack>
        {status ? (
          <Text color="fg.subtle" fontSize="md" px="3" py="2" role="status">
            {status}
          </Text>
        ) : null}
        <Box
          aria-label={t('cliptags.prompt.suggestions')}
          display={candidates.length > 0 ? undefined : 'none'}
          id={listboxId}
          minH="0"
          overflowY="auto"
          py="1"
          ref={listRef}
          role="listbox"
          onScroll={handleScroll}
        >
          {candidates.map((candidate, index) => (
            <TagOption
              key={candidate.id}
              candidate={candidate}
              id={`${optionIdPrefix}${index}`}
              isActive={index === activeIndex}
              onSelect={onSelect}
            />
          ))}
          {isFetchingMore ? (
            <Text color="fg.subtle" fontSize="md" px="3" py="1" role="status">
              {t('cliptags.prompt.loadingMore')}
            </Text>
          ) : null}
        </Box>
      </Stack>
    </Portal>
  );
};

const CategoryChip = ({
  isActive,
  label,
  onSelect,
  value,
}: {
  isActive: boolean;
  label: string;
  value: ClipTagCategory | null;
  onSelect: (category: ClipTagCategory | null) => void;
}) => {
  const handleMouseDown = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      // Keep focus in the prompt, which is what keeps the suggestions open.
      event.preventDefault();

      if (event.button === 0) {
        onSelect(value);
      }
    },
    [onSelect, value]
  );

  return (
    <chakra.button
      aria-pressed={isActive}
      bg={isActive ? 'bg.hover' : undefined}
      borderColor={isActive ? 'border.emphasized' : 'transparent'}
      borderRadius="sm"
      borderWidth="1px"
      color={isActive ? 'fg' : 'fg.muted'}
      cursor="pointer"
      fontSize="md"
      px="2"
      tabIndex={-1}
      type="button"
      _hover={OPTION_HOVER_CSS}
      onMouseDown={handleMouseDown}
    >
      {label}
    </chakra.button>
  );
};

const TagOption = ({
  candidate,
  id,
  isActive,
  onSelect,
}: {
  candidate: ClipTagCandidate;
  id: string;
  isActive: boolean;
  onSelect: (candidate: ClipTagCandidate) => void;
}) => {
  const { t } = useTranslation();
  const handleMouseDown = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }

      event.preventDefault();
      onSelect(candidate);
    },
    [candidate, onSelect]
  );
  const popularity = useMemo(
    () => (candidate.popularity === null ? null : POPULARITY_FORMAT.format(candidate.popularity)),
    [candidate.popularity]
  );

  return (
    <HStack
      aria-selected={isActive}
      bg={isActive ? 'bg.hover' : undefined}
      color="fg"
      fontSize="md"
      gap="2"
      id={id}
      px="2"
      py="1"
      role="option"
      tabIndex={-1}
      _hover={OPTION_HOVER_CSS}
      onMouseDown={handleMouseDown}
    >
      <Box
        aria-hidden
        bg="colorPalette.solid"
        borderRadius="full"
        colorPalette={CLIP_TAG_TYPE_PALETTE[candidate.type]}
        flexShrink={0}
        h="2"
        w="2"
      />
      <MiddleTruncate as="span" text={candidate.renderedContent} />
      <VisuallyHidden>{t(`cliptags.types.${candidate.type}`)}</VisuallyHidden>
      {popularity ? (
        <Text color="fg.subtle" flexShrink={0} fontSize="md" ml="auto">
          {popularity}
        </Text>
      ) : null}
    </HStack>
  );
};

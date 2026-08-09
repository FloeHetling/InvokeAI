import type { ChakraProps, ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Box, Combobox, Flex, FormControl, HStack, IconButton, Spinner, Text, Tooltip } from '@invoke-ai/ui-library';
import type {
  GroupBase,
  InputActionMeta,
  MenuListProps,
  OptionProps,
  SelectComponentsConfig,
} from 'chakra-react-select';
import { getOverlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { useClipTagAutocomplete } from 'features/clipTagAutocomplete/hooks/useClipTagAutocomplete';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import {
  Children,
  createContext,
  isValidElement,
  type KeyboardEventHandler,
  memo,
  type MouseEventHandler,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { PiPaintBrush, PiShapes, PiUser } from 'react-icons/pi';
import type { VirtuosoHandle } from 'react-virtuoso';
import { Virtuoso } from 'react-virtuoso';
import type { CtaAutocompleteFilter } from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';

type CtaAutocompleteCandidate = S['CtaAutocompleteCandidate'];

type CtaComboboxOption = ComboboxOption & {
  candidate: CtaAutocompleteCandidate;
};

type CtaBrowseHintOption = ComboboxOption & {
  isCtaBrowseHint: true;
};

type ClipTagAutocompleteSelectProps = {
  modelId: string | null;
  onSelect: (renderedContent: string, keepOpen?: boolean) => void;
  onClose: () => void;
};

type CtaMenuContextValue = {
  filter: CtaAutocompleteFilter | null;
  hasMore: boolean;
  loadMore: () => void;
};

type CtaOptionProps = OptionProps<ComboboxOption, false, GroupBase<ComboboxOption>>;
type CtaMenuListProps = MenuListProps<ComboboxOption, false, GroupBase<ComboboxOption>>;

const CTA_FILTER_SESSION_KEY = 'invokeai.clipTagAutocomplete.filter';
const CTA_FILTERS: CtaAutocompleteFilter[] = ['artist', 'character', 'other'];
const CTA_UNFILTERED_RESULT_LIMIT = 20;
const CTA_BROWSE_HINT_VALUE = '__cta_browse_hint__';
const CTA_OPTION_CACHE = new WeakMap<CtaAutocompleteCandidate, CtaComboboxOption>();
const CtaMenuContext = createContext<CtaMenuContextValue>({
  filter: null,
  hasMore: false,
  loadMore: () => undefined,
});

const TAG_TYPE_COLORS: Record<CtaAutocompleteCandidate['tag_type'], string> = {
  general: 'blue.500',
  artist: 'red.500',
  copyright: 'purple.500',
  character: 'green.500',
  meta: 'yellow.500',
  other: 'gray.500',
};

const CTA_FILTERED_SCROLLBARS_PARAMS = getOverlayScrollbarsParams({
  overflowX: 'hidden',
  overflowY: 'scroll',
  visibility: 'visible',
});
const CTA_VIRTUOSO_STYLE = { height: '100%', width: '100%' };

const isCtaAutocompleteFilter = (value: string | null): value is CtaAutocompleteFilter =>
  value !== null && CTA_FILTERS.includes(value as CtaAutocompleteFilter);

const readStoredFilter = (): CtaAutocompleteFilter | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    const value = window.sessionStorage.getItem(CTA_FILTER_SESSION_KEY);
    return isCtaAutocompleteFilter(value) ? value : null;
  } catch {
    return null;
  }
};

const storeFilter = (filter: CtaAutocompleteFilter | null): void => {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    if (filter === null) {
      window.sessionStorage.removeItem(CTA_FILTER_SESSION_KEY);
    } else {
      window.sessionStorage.setItem(CTA_FILTER_SESSION_KEY, filter);
    }
  } catch {
    // Session storage can be unavailable in hardened/private browser contexts.
  }
};

/**
 * react-select owns Home/End while the menu is open, but for CTA the search input should retain
 * normal single-line text-editing semantics for those keys and their Shift/Ctrl combinations.
 * Calling preventDefault() here also tells react-select to skip its own list-navigation handler.
 */
const handleCtaInputBoundaryKeyDown: KeyboardEventHandler<HTMLDivElement> = (event) => {
  if ((event.key !== 'Home' && event.key !== 'End') || !(event.target instanceof HTMLInputElement)) {
    return;
  }

  const input = event.target;
  const target = event.key === 'Home' ? 0 : input.value.length;
  event.preventDefault();

  if (!event.shiftKey) {
    input.setSelectionRange(target, target);
    return;
  }

  const selectionStart = input.selectionStart ?? 0;
  const selectionEnd = input.selectionEnd ?? selectionStart;
  const anchor = input.selectionDirection === 'backward' ? selectionEnd : selectionStart;

  input.setSelectionRange(Math.min(anchor, target), Math.max(anchor, target), target < anchor ? 'backward' : 'forward');
};

const isCtaComboboxOption = (option: ComboboxOption): option is CtaComboboxOption => 'candidate' in option;

const isCtaBrowseHintOption = (option: ComboboxOption): option is CtaBrowseHintOption =>
  'isCtaBrowseHint' in option && option.isCtaBrowseHint === true;

const getCtaComboboxOption = (candidate: CtaAutocompleteCandidate): CtaComboboxOption => {
  const cached = CTA_OPTION_CACHE.get(candidate);
  if (cached) {
    return cached;
  }
  const option: CtaComboboxOption = {
    label: candidate.canonical_content,
    value: `${candidate.id}:${candidate.tag_type}`,
    candidate,
  };
  CTA_OPTION_CACHE.set(candidate, option);
  return option;
};

/**
 * CTA's filtered combobox components intentionally use only primitives re-exported by
 * @invoke-ai/ui-library. The app also depends on chakra-react-select directly, and using
 * its runtime chakraComponents here can bind these children to a different Chakra runtime
 * than the Combobox/provider owned by the UI library.
 */
const CtaOption = ({ innerProps: optionInnerProps, isFocused, isDisabled, children }: CtaOptionProps) => {
  const innerProps = {
    ...optionInnerProps,
    onMouseMove: undefined,
    onMouseOver: undefined,
  };

  return (
    <Flex
      {...innerProps}
      w="full"
      alignItems="center"
      p={1}
      px={4}
      borderRadius="sm"
      bg={isFocused ? 'base.700' : 'base.800'}
      cursor={isDisabled ? 'not-allowed' : 'pointer'}
      opacity={isDisabled ? 0.5 : 1}
      _hover={isDisabled ? undefined : { bg: 'base.700' }}
      data-focus={isFocused ? true : undefined}
    >
      {children}
    </Flex>
  );
};

const CtaVirtualizedMenuList = (props: CtaMenuListProps) => {
  const { children, innerProps, innerRef } = props;
  const { focusedOption } = props;
  const { hasMore, loadMore } = useContext(CtaMenuContext);
  const items = useMemo(() => Children.toArray(children), [children]);
  const menuHeight = Math.min(props.maxHeight, items.length * 36);
  const menuRootRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const lastScrolledFocusedOptionRef = useRef<ComboboxOption | null>(null);
  const [initializeScrollbars, getScrollbarsInstance] = useOverlayScrollbars(CTA_FILTERED_SCROLLBARS_PARAMS);

  const focusedIndex = useMemo(
    () => items.findIndex((item) => isValidElement<CtaOptionProps>(item) && item.props.data === focusedOption),
    [focusedOption, items]
  );

  const handleScrollerRef = useCallback((node: HTMLElement | Window | null) => {
    setViewport(node instanceof HTMLDivElement ? node : null);
  }, []);

  useEffect(() => {
    const root = menuRootRef.current;
    if (root && viewport) {
      initializeScrollbars({
        target: root,
        elements: {
          viewport,
        },
      });
    }

    return () => {
      getScrollbarsInstance()?.destroy();
    };
  }, [getScrollbarsInstance, initializeScrollbars, viewport]);

  useEffect(() => {
    if (typeof innerRef === 'function') {
      innerRef(viewport);
    }
  }, [innerRef, viewport]);

  useEffect(() => {
    if (focusedIndex < 0 || !focusedOption) {
      lastScrolledFocusedOptionRef.current = null;
      return;
    }

    if (lastScrolledFocusedOptionRef.current !== focusedOption && virtuosoRef.current) {
      lastScrolledFocusedOptionRef.current = focusedOption;
      virtuosoRef.current.scrollIntoView({ index: focusedIndex, behavior: 'auto' });
    }

    if (hasMore && focusedIndex >= items.length - 2) {
      loadMore();
    }
  }, [focusedIndex, focusedOption, hasMore, items.length, loadMore]);

  const handleEndReached = useCallback(() => {
    if (hasMore) {
      loadMore();
    }
  }, [hasMore, loadMore]);

  const computeItemKey = useCallback((index: number, item: (typeof items)[number]) => {
    if (isValidElement(item) && item.key !== null) {
      return item.key;
    }
    return index;
  }, []);

  return (
    <Box
      {...innerProps}
      ref={menuRootRef}
      data-overlayscrollbars-initialize=""
      h={`${menuHeight}px`}
      maxH={`${props.maxHeight}px`}
      position="relative"
      color="base.150"
      bg="base.800"
      shadow="dark-lg"
      borderRadius="md"
      p={1}
    >
      <Virtuoso
        ref={virtuosoRef}
        style={CTA_VIRTUOSO_STYLE}
        scrollerRef={handleScrollerRef}
        data={items}
        computeItemKey={computeItemKey}
        increaseViewportBy={{ top: 120, bottom: 240 }}
        endReached={handleEndReached}
        itemContent={(_index, item) => item}
      />
    </Box>
  );
};

const CtaPlainMenuList = (props: CtaMenuListProps) => (
  <Box
    {...props.innerProps}
    ref={(node) => {
      if (typeof props.innerRef === 'function') {
        props.innerRef(node);
      }
    }}
    maxH={`${props.maxHeight}px`}
    overflowY="auto"
    position="relative"
    color="base.150"
    bg="base.800"
    shadow="dark-lg"
    borderRadius="md"
    p={1}
  >
    {props.children}
  </Box>
);

const CtaMenuList = (props: CtaMenuListProps) => {
  const { filter } = useContext(CtaMenuContext);

  if (filter === null || Children.count(props.children) < 2) {
    return <CtaPlainMenuList {...props} />;
  }

  return <CtaVirtualizedMenuList {...props} />;
};

const CtaLoadingIndicator = () => (
  <Flex w="full" h="full" alignItems="center" justifyContent="center">
    <Spinner size="xs" color="base.300" />
  </Flex>
);

const CTA_LOADING_COMPONENTS: SelectComponentsConfig<ComboboxOption, false, GroupBase<ComboboxOption>> = {
  LoadingIndicator: CtaLoadingIndicator,
};
const CTA_FILTERED_COMBOBOX_COMPONENTS: SelectComponentsConfig<ComboboxOption, false, GroupBase<ComboboxOption>> = {
  ...CTA_LOADING_COMPONENTS,
  Option: CtaOption,
  MenuList: CtaMenuList,
};

export const ClipTagAutocompleteSelect = memo(({ modelId, onSelect, onClose }: ClipTagAutocompleteSelectProps) => {
  const { t } = useTranslation();
  const [inputValue, setInputValue] = useState('');
  const keepOpenOnSelectRef = useRef(false);
  const [filter, setFilter] = useState<CtaAutocompleteFilter | null>(readStoredFilter);
  const { candidates, isSearching, isLoadingMore, hasMore, loadMore, normalizedQuery, isAvailable, isStatusLoading } =
    useClipTagAutocomplete({
      query: inputValue,
      modelId,
      filter,
    });

  const options = useMemo<ComboboxOption[]>(() => {
    const result: ComboboxOption[] = candidates.map(getCtaComboboxOption);

    if (filter === null && candidates.length === CTA_UNFILTERED_RESULT_LIMIT) {
      const browseHint: CtaBrowseHintOption = {
        label: t('cta.unfilteredResultLimitHint', { limit: CTA_UNFILTERED_RESULT_LIMIT }),
        value: CTA_BROWSE_HINT_VALUE,
        isCtaBrowseHint: true,
      };
      result.push(browseHint);
    }

    return result;
  }, [candidates, filter, t]);
  const menuContext = useMemo<CtaMenuContextValue>(() => ({ filter, hasMore, loadMore }), [filter, hasMore, loadMore]);
  const isLoading = isSearching || isLoadingMore || isStatusLoading;

  const handleFilterToggle = useCallback(
    (nextFilter: CtaAutocompleteFilter) => {
      const value = filter === nextFilter ? null : nextFilter;
      setFilter(value);
      storeFilter(value);
    },
    [filter]
  );

  const handleFilterMouseDown = useCallback<MouseEventHandler<HTMLButtonElement>>((event) => {
    event.preventDefault();
  }, []);

  const handleClickCapture = useCallback<MouseEventHandler<HTMLDivElement>>((event) => {
    keepOpenOnSelectRef.current = event.ctrlKey || event.metaKey;

    // react-select selects options from the click handler after this capture phase. Clear the
    // modifier on the next task so unrelated keyboard selections cannot inherit a stale click.
    setTimeout(() => {
      keepOpenOnSelectRef.current = false;
    }, 0);
  }, []);

  const handleChange = useCallback<ComboboxOnChange>(
    (value) => {
      if (!value || !isCtaComboboxOption(value)) {
        return;
      }
      const keepOpen = keepOpenOnSelectRef.current;
      keepOpenOnSelectRef.current = false;
      onSelect(value.candidate.rendered_content, keepOpen);
    },
    [onSelect]
  );

  const handleInputChange = useCallback((value: string, meta: InputActionMeta) => {
    if (meta.action === 'input-change') {
      setInputValue(value);
    }
    return value;
  }, []);

  const noOptionsMessage = useCallback(() => {
    if (isStatusLoading) {
      return t('common.loading');
    }
    if (!isAvailable) {
      return t('cta.unavailable');
    }
    if (normalizedQuery.length < 2) {
      return t('cta.typeAtLeastTwoCharacters');
    }
    return t('cta.noMatchingTags');
  }, [isAvailable, isStatusLoading, normalizedQuery.length, t]);

  const formatOptionLabel = useCallback((option: ComboboxOption) => {
    if (isCtaBrowseHintOption(option)) {
      return (
        <Flex as="span" w="full" mt={1} pt={2} borderTopWidth="1px" borderColor="base.700">
          <Text as="span" fontSize="xs" color="base.300">
            {option.label}
          </Text>
        </Flex>
      );
    }

    if (!isCtaComboboxOption(option)) {
      return option.label;
    }

    const { candidate } = option;
    return (
      <Flex as="span" alignItems="center" gap={2} minW={0} w="full">
        <Text
          as="span"
          fontSize="sm"
          color={TAG_TYPE_COLORS[candidate.tag_type]}
          overflow="hidden"
          textOverflow="ellipsis"
        >
          {candidate.canonical_content}
        </Text>
        {candidate.popularity !== null && candidate.popularity !== undefined && (
          <Text as="span" fontSize="xs" color="gray.400" ml="auto" flexShrink={0}>
            {candidate.popularity.toLocaleString()}
          </Text>
        )}
      </Flex>
    );
  }, []);

  return (
    <CtaMenuContext.Provider value={menuContext}>
      <FormControl onClickCapture={handleClickCapture}>
        <HStack spacing={1} px={1} pb={1}>
          <Tooltip label={t('cta.filterArtistsAndCopyrights')}>
            <IconButton
              aria-label={t('cta.filterArtistsAndCopyrights')}
              aria-pressed={filter === 'artist'}
              icon={<PiPaintBrush />}
              size="sm"
              variant={filter === 'artist' ? 'solid' : 'ghost'}
              onMouseDown={handleFilterMouseDown}
              onClick={() => handleFilterToggle('artist')}
            />
          </Tooltip>
          <Tooltip label={t('cta.filterCharacters')}>
            <IconButton
              aria-label={t('cta.filterCharacters')}
              aria-pressed={filter === 'character'}
              icon={<PiUser />}
              size="sm"
              variant={filter === 'character' ? 'solid' : 'ghost'}
              onMouseDown={handleFilterMouseDown}
              onClick={() => handleFilterToggle('character')}
            />
          </Tooltip>
          <Tooltip label={t('cta.filterOtherTags')}>
            <IconButton
              aria-label={t('cta.filterOtherTags')}
              aria-pressed={filter === 'other'}
              icon={<PiShapes />}
              size="sm"
              variant={filter === 'other' ? 'solid' : 'ghost'}
              onMouseDown={handleFilterMouseDown}
              onClick={() => handleFilterToggle('other')}
            />
          </Tooltip>
        </HStack>
        <Combobox
          placeholder={t('cta.searchTags')}
          defaultMenuIsOpen
          autoFocus
          value={null}
          inputValue={inputValue}
          options={options}
          isLoading={isLoading}
          filterOption={() => true}
          isOptionDisabled={isCtaBrowseHintOption}
          tabSelectsValue={false}
          closeMenuOnSelect={false}
          noOptionsMessage={noOptionsMessage}
          onInputChange={handleInputChange}
          onKeyDown={handleCtaInputBoundaryKeyDown}
          onChange={handleChange}
          onMenuClose={onClose}
          formatOptionLabel={formatOptionLabel}
          {...(filter === null
            ? isLoading
              ? { components: CTA_LOADING_COMPONENTS }
              : {}
            : { components: CTA_FILTERED_COMBOBOX_COMPONENTS })}
          data-testid="clip-tag-autocomplete"
          sx={selectStyles}
        />
      </FormControl>
    </CtaMenuContext.Provider>
  );
});

ClipTagAutocompleteSelect.displayName = 'ClipTagAutocompleteSelect';

const selectStyles: ChakraProps['sx'] = {
  w: 'full',
};

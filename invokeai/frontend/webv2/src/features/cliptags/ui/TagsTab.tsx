import type { ClipTag, ClipTagFilter, ClipTagSet, ClipTagType } from '@features/cliptags/core/types';
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ListRowProps } from '@platform/ui/list/List';

import { Box, HStack, Icon, Input, InputGroup, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { clipTagsInfiniteOptions, clipTagSetsOptions } from '@features/cliptags/data/queries';
import { formatCount } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { ListSelectionBar } from '@platform/ui/list/ListSelectionBar';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { keepPreviousData, useInfiniteQuery, useQuery } from '@tanstack/react-query';
import {
  ChevronDownIcon,
  FolderMinusIcon,
  FolderPlusIcon,
  SearchIcon,
  TagIcon,
  TagsIcon,
  Trash2Icon,
  UploadIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { BulkOperationDialog, type BulkOperationKind } from './BulkOperationDialog';
import { OptionSelect, type Option } from './OptionSelect';
import { TagEditorDialog } from './TagEditorDialog';
import {
  getSelectAllState,
  getSelectedCount,
  isIdSelectionFull,
  isSameTagFilter,
  isTagSelected,
  NO_TAG_SELECTION,
  pruneTagSelection,
  selectAllMatching,
  toBulkSelection,
  toggleLoadedSelection,
  toggleTagSelection,
  type TagSelection,
} from './tagSelection';
import { isClipTagType, TagTypeBadge, useTagTypeOptions } from './tagTypes';

const SEARCH_DEBOUNCE_MS = 300;
const ANY_TYPE = '';
const ANY_SET = '';
const UNCATEGORIZED = '__uncategorized__';
const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;
const EMPTY_TAGS: readonly ClipTag[] = [];
const EMPTY_TAG_SETS: readonly ClipTagSet[] = [];
const MENU_POSITION = { placement: 'bottom-end' } as const;
const getTagId = (tag: ClipTag): string => tag.id;

export interface TagsTabProps {
  /** Tag set the list starts filtered to, e.g. when the user came from the tag set's row. */
  initialTagSetId?: string;
  onImport: () => void;
}

/** Browse, search, edit and bulk-edit the tag database. */
export const TagsTab = ({ initialTagSetId, onImport }: TagsTabProps) => {
  const { t } = useTranslation();
  const typeOptions = useTagTypeOptions();
  const tagSetsQuery = useQuery(clipTagSetsOptions());
  const tagSets = tagSetsQuery.data;
  const [search, setSearch] = useState('');
  const [querySearch, setQuerySearch] = useState('');
  const searchTimerRef = useRef<number | undefined>(undefined);
  const [typeFilter, setTypeFilter] = useState<ClipTagType | typeof ANY_TYPE>(ANY_TYPE);
  const [setFilter, setSetFilter] = useState<string>(initialTagSetId ?? ANY_SET);
  const [editingTag, setEditingTag] = useState<ClipTag | null>(null);
  const [bulkKind, setBulkKind] = useState<BulkOperationKind | null>(null);

  useMountEffect(() => () => window.clearTimeout(searchTimerRef.current));

  // A filter on a tag set that no longer exists (it was deleted) means no filter.
  const effectiveSetFilter =
    tagSets !== undefined &&
    setFilter !== ANY_SET &&
    setFilter !== UNCATEGORIZED &&
    !tagSets.some((s) => s.id === setFilter)
      ? ANY_SET
      : setFilter;
  const filter = useMemo<ClipTagFilter>(
    () => ({
      search: querySearch.trim() || undefined,
      tagSetId: effectiveSetFilter !== ANY_SET && effectiveSetFilter !== UNCATEGORIZED ? effectiveSetFilter : undefined,
      tagType: typeFilter === ANY_TYPE ? undefined : typeFilter,
      uncategorized: effectiveSetFilter === UNCATEGORIZED ? true : undefined,
    }),
    [effectiveSetFilter, querySearch, typeFilter]
  );
  // A selection belongs to the filter it was made under. Any change to the effective filter, including the fallback
  // when a tag set was deleted elsewhere, drops it: it must never act on rows other than the ones the user saw.
  const [owned, setOwned] = useState<{ filter: ClipTagFilter; selection: TagSelection }>({
    filter,
    selection: NO_TAG_SELECTION,
  });
  const isOwnedByFilter = isSameTagFilter(owned.filter, filter);

  if (!isOwnedByFilter) {
    setOwned({ filter, selection: NO_TAG_SELECTION });
  }

  const selection = isOwnedByFilter ? owned.selection : NO_TAG_SELECTION;
  const setSelection = useCallback(
    (next: TagSelection) => setOwned((current) => ({ ...current, selection: next })),
    []
  );
  const hasFilter = filter.search !== undefined || filter.tagType !== undefined || effectiveSetFilter !== ANY_SET;

  // Rows from the previous filter stay on screen, inert, while the next ones load.
  const query = useInfiniteQuery({ ...clipTagsInfiniteOptions(filter), placeholderData: keepPreviousData });
  const { fetchNextPage, hasNextPage, isFetchingNextPage } = query;
  const tags = useMemo(() => {
    if (!query.data) {
      return EMPTY_TAGS;
    }

    // A tag can move between pages while the user loads them; show it once.
    return [...new Map(query.data.pages.flatMap((page) => page.items).map((tag) => [tag.id, tag])).values()];
  }, [query.data]);
  const total = query.data?.pages.at(-1)?.total ?? 0;
  const isListBusy = query.isPlaceholderData || search !== querySearch;
  const loadedIds = useMemo(() => tags.map(getTagId), [tags]);
  const rows = useMemo(() => listRowsFromItems(tags, getTagId), [tags]);
  const effectiveSelection = useMemo(() => pruneTagSelection(selection, loadedIds), [loadedIds, selection]);
  const selectedCount = getSelectedCount(effectiveSelection);
  const selectAllState = getSelectAllState(effectiveSelection, loadedIds);
  // Offered once every row that can be picked by id is, and whenever more tags match than are selected.
  const canSelectMatching =
    effectiveSelection.mode === 'ids' &&
    total > selectedCount &&
    (isIdSelectionFull(effectiveSelection) || (selectAllState === true && total > loadedIds.length));

  const tagSetFilterOptions = useMemo<Option[]>(
    () => [
      { label: t('cliptags.manager.tags.allTagSets'), value: ANY_SET },
      { label: t('cliptags.manager.tags.uncategorized'), value: UNCATEGORIZED },
      ...(tagSets ?? []).map((tagSet) => ({ label: tagSet.name, value: tagSet.id })),
    ],
    [t, tagSets]
  );
  const typeFilterOptions = useMemo<Option[]>(
    () => [{ label: t('cliptags.manager.tags.allTypes'), value: ANY_TYPE }, ...typeOptions],
    [t, typeOptions]
  );

  const commitSearch = useCallback((value: string) => {
    window.clearTimeout(searchTimerRef.current);
    setQuerySearch(value);
  }, []);
  const handleSearchChange = useCallback(
    (event: { currentTarget: { value: string } }) => {
      const value = event.currentTarget.value;

      setSearch(value);
      window.clearTimeout(searchTimerRef.current);
      searchTimerRef.current = window.setTimeout(() => commitSearch(value), SEARCH_DEBOUNCE_MS);
    },
    [commitSearch]
  );
  const handleTypeFilterChange = useCallback((value: string) => {
    setTypeFilter(isClipTagType(value) ? value : ANY_TYPE);
  }, []);
  const handleSetFilterChange = useCallback((value: string) => setSetFilter(value), []);
  const clearFilters = useCallback(() => {
    setSearch('');
    commitSearch('');
    setTypeFilter(ANY_TYPE);
    setSetFilter(ANY_SET);
  }, [commitSearch]);

  const handleToggleAll = useCallback(
    () => setSelection(toggleLoadedSelection(effectiveSelection, loadedIds)),
    [effectiveSelection, loadedIds, setSelection]
  );
  const handleToggleTag = useCallback(
    (id: string) => setSelection(toggleTagSelection(effectiveSelection, id, loadedIds)),
    [effectiveSelection, loadedIds, setSelection]
  );
  const clearSelection = useCallback(() => setSelection(NO_TAG_SELECTION), [setSelection]);
  const handleSelectMatching = useCallback(
    () => setSelection(selectAllMatching(total, filter)),
    [filter, setSelection, total]
  );
  const getBulkSelection = useCallback(() => toBulkSelection(effectiveSelection), [effectiveSelection]);
  const closeEditor = useCallback(() => setEditingTag(null), []);
  const closeBulk = useCallback(() => setBulkKind(null), []);
  const finishBulk = useCallback(() => {
    setSelection(NO_TAG_SELECTION);
    setBulkKind(null);
  }, [setSelection]);
  const loadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) {
      void fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);
  const retry = useCallback(() => void query.refetch(), [query]);

  const renderItem = (tag: ClipTag, rowProps: ListRowProps) => (
    <ListItem
      {...rowProps}
      checkLabel={t('cliptags.manager.tags.selectTag', { name: tag.content })}
      description={
        tag.popularity === null
          ? t('cliptags.manager.tags.popularityUnknown')
          : t('cliptags.manager.tags.popularity', { value: formatCount(tag.popularity) })
      }
      isActive={editingTag?.id === tag.id}
      isBusy={isListBusy}
      isChecked={isTagSelected(effectiveSelection, tag.id)}
      title={tag.content}
      titleTruncate="end"
      trailing={<TagTypeBadge type={tag.type} />}
      onCheckedChange={() => handleToggleTag(tag.id)}
      onPress={() => setEditingTag(tag)}
    />
  );

  return (
    <Stack flex="1" gap="3" minH="0">
      <HStack flexWrap="wrap" gap="2">
        <InputGroup flex="1 1 12rem" minW="0" startElement={SEARCH_ICON}>
          <Input
            aria-label={t('cliptags.manager.tags.searchLabel')}
            autoComplete="off"
            placeholder={t('cliptags.manager.tags.searchPlaceholder')}
            value={search}
            onChange={handleSearchChange}
          />
        </InputGroup>
        <Box flex="0 1 9rem" minW="7rem">
          <OptionSelect
            label={t('cliptags.manager.tags.typeFilter')}
            options={typeFilterOptions}
            value={typeFilter}
            onChange={handleTypeFilterChange}
          />
        </Box>
        <Box flex="0 1 12rem" minW="8rem">
          <OptionSelect
            label={t('cliptags.manager.tags.tagSetFilter')}
            options={tagSetFilterOptions}
            value={effectiveSetFilter}
            onChange={handleSetFilterChange}
          />
        </Box>
      </HStack>
      <Stack flex="1" gap="0" minH="0">
        <ListSelectionBar
          checked={selectAllState}
          isDisabled={tags.length === 0 || isListBusy}
          label={t('cliptags.manager.tags.selectAll')}
          summary={
            selectedCount > 0
              ? t('cliptags.manager.tags.selected', { count: selectedCount, total: formatCount(selectedCount) })
              : query.data
                ? t('cliptags.manager.tags.total', { count: total, total: formatCount(total) })
                : ''
          }
          onCheckedChange={handleToggleAll}
        >
          {selectedCount > 0 ? <BulkActionsMenu disabled={isListBusy} onSelect={setBulkKind} /> : null}
        </ListSelectionBar>
        {canSelectMatching || effectiveSelection.mode === 'matching' ? (
          <HStack
            bg="bg.subtle"
            borderBottomWidth="1px"
            borderColor="border.subtle"
            flexWrap="wrap"
            gap="2"
            px="3"
            py="1"
          >
            <Text color="fg.muted" fontSize="xs">
              {effectiveSelection.mode === 'matching'
                ? t('cliptags.manager.tags.allMatchingSelected', { total: formatCount(total) })
                : isIdSelectionFull(effectiveSelection)
                  ? t('cliptags.manager.tags.selectionLimit', { total: formatCount(selectedCount) })
                  : t('cliptags.manager.tags.pageSelected', {
                      count: loadedIds.length,
                      total: formatCount(loadedIds.length),
                    })}
            </Text>
            <Button
              colorPalette="accent"
              disabled={isListBusy}
              size="xs"
              variant="ghost"
              onClick={effectiveSelection.mode === 'matching' ? clearSelection : handleSelectMatching}
            >
              {effectiveSelection.mode === 'matching'
                ? t('cliptags.manager.tags.clearSelection')
                : t('cliptags.manager.tags.selectAllMatching', { total: formatCount(total) })}
            </Button>
          </HStack>
        ) : null}
        <List
          activeKey={editingTag?.id ?? null}
          density="comfortable"
          emptyState={
            <EmptyState
              description={
                hasFilter
                  ? t('cliptags.manager.tags.noMatchesDescription')
                  : t('cliptags.manager.tags.emptyDescription')
              }
              icon={<Icon as={hasFilter ? SearchIcon : TagsIcon} />}
              title={hasFilter ? t('cliptags.manager.tags.noMatches') : t('cliptags.manager.tags.empty')}
            >
              {hasFilter ? (
                <Button variant="outline" onClick={clearFilters}>
                  {t('cliptags.manager.tags.clearFilters')}
                </Button>
              ) : (
                <Button onClick={onImport}>
                  <UploadIcon />
                  {t('cliptags.manager.import.open')}
                </Button>
              )}
            </EmptyState>
          }
          errorState={
            <EmptyState
              danger
              description={query.isError ? getApiErrorMessage(query.error, '') : null}
              title={t('cliptags.manager.tags.loadFailed')}
            >
              <Button variant="outline" onClick={retry}>
                {t('common.retry')}
              </Button>
            </EmptyState>
          }
          isBusy={isListBusy}
          label={t('cliptags.manager.tags.listLabel')}
          renderItem={renderItem}
          rows={rows}
          status={query.isPending ? 'loading' : query.isError && tags.length === 0 ? 'error' : 'ready'}
        />
        {tags.length > 0 ? (
          <HStack borderColor="border.subtle" borderTopWidth="1px" flexShrink={0} justify="center" minH="8">
            <Text aria-live="polite" color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
              {t('common.countOfTotal', { count: formatCount(tags.length), total: formatCount(total) })}
            </Text>
            {hasNextPage ? (
              <Button
                aria-disabled={isFetchingNextPage || isListBusy}
                loading={isFetchingNextPage}
                size="sm"
                variant="ghost"
                onClick={isListBusy ? undefined : loadMore}
              >
                {t('cliptags.manager.tags.loadMore')}
              </Button>
            ) : null}
          </HStack>
        ) : null}
      </Stack>
      <TagEditorDialog tag={editingTag} tagSets={tagSets ?? EMPTY_TAG_SETS} onClose={closeEditor} />
      <BulkOperationDialog
        count={selectedCount}
        getSelection={getBulkSelection}
        kind={bulkKind}
        tagSets={tagSets ?? EMPTY_TAG_SETS}
        onClose={closeBulk}
        onDone={finishBulk}
      />
    </Stack>
  );
};

const BulkActionsMenu = ({
  disabled,
  onSelect,
}: {
  disabled: boolean;
  onSelect: (kind: BulkOperationKind) => void;
}) => {
  const { t } = useTranslation();

  return (
    <Menu.Root positioning={MENU_POSITION}>
      <Menu.Trigger asChild>
        <Button aria-disabled={disabled} size="sm" variant="ghost">
          {t('cliptags.manager.bulk.actions')}
          <Icon as={ChevronDownIcon} boxSize="3" />
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="12rem">
            <MenuActionItem
              disabled={disabled}
              icon={FolderPlusIcon}
              label={t('cliptags.manager.bulk.menu.add_to_set')}
              value="add_to_set"
              onSelect={() => onSelect('add_to_set')}
            />
            <MenuActionItem
              disabled={disabled}
              icon={FolderMinusIcon}
              label={t('cliptags.manager.bulk.menu.remove_from_set')}
              value="remove_from_set"
              onSelect={() => onSelect('remove_from_set')}
            />
            <MenuActionItem
              disabled={disabled}
              icon={TagIcon}
              label={t('cliptags.manager.bulk.menu.set_type')}
              value="set_type"
              onSelect={() => onSelect('set_type')}
            />
            <MenuActionItem
              disabled={disabled}
              icon={Trash2Icon}
              label={t('cliptags.manager.bulk.menu.delete')}
              tone="danger"
              value="delete"
              onSelect={() => onSelect('delete')}
            />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

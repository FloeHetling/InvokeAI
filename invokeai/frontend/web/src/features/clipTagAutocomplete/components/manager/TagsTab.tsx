import { Button, Checkbox, Flex, HStack, Input, Select, Text, useToast, VStack } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useCtaBeforeUnload } from 'features/clipTagAutocomplete/hooks/useCtaBeforeUnload';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useBulkMutateCtaTagsMutation,
  useDeleteCtaTagMutation,
  useGetCtaTagQuery,
  useListCtaTagSetsQuery,
  useListCtaTagsQuery,
  useUpdateCtaTagMutation,
} from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';
import { useDebounce } from 'use-debounce';

import { CtaManagerOverlay } from './CtaManagerOverlay';
import { CtaMutationProgressModal } from './CtaMutationProgressModal';

type CtaTagDTO = S['CtaTagDTO'];
type CtaTagFilter = S['CtaTagFilter'];
type CtaTagType = 'general' | 'artist' | 'copyright' | 'character' | 'meta' | 'other';
type CtaBulkOperation =
  | { type: 'add_to_set'; tag_set_id: string }
  | { type: 'remove_from_set'; tag_set_id: string }
  | { type: 'set_type'; tag_type: CtaTagType }
  | { type: 'delete' };

type SelectionMode = { mode: 'ids'; ids: Set<string> } | { mode: 'filter'; filter: CtaTagFilter; total: number };

const TAG_TYPE_COLORS: Record<string, string> = {
  general: 'blue.500',
  artist: 'red.500',
  copyright: 'purple.500',
  character: 'green.500',
  meta: 'yellow.500',
  other: 'gray.500',
};

const TAG_TYPES: CtaTagType[] = ['general', 'artist', 'copyright', 'character', 'meta', 'other'];

type Props = {
  initialTagSetId?: string;
  onImport: () => void;
  onMutationBusyChange: (isBusy: boolean) => void;
};

export const TagsTab = memo(({ initialTagSetId, onImport, onMutationBusyChange }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const [search, setSearch] = useState('');
  const [debouncedSearch] = useDebounce(search, 300);
  const [tagTypeFilter, setTagTypeFilter] = useState('');
  const [tagSetFilter, setTagSetFilter] = useState(initialTagSetId ?? '');
  const [cursor, setCursor] = useState<string | undefined>();
  const [cursorStack, setCursorStack] = useState<(string | undefined)[]>([]);
  const [selection, setSelection] = useState<SelectionMode>({ mode: 'ids', ids: new Set() });
  const [selectedTagId, setSelectedTagId] = useState<string | null>(null);
  const [bulkTagSetId, setBulkTagSetId] = useState('');
  const [bulkTagType, setBulkTagType] = useState<CtaTagType>('general');
  const [bulkMutate, { isLoading: isBulkMutating }] = useBulkMutateCtaTagsMutation();
  const [updateTag, { isLoading: isUpdatingTag }] = useUpdateCtaTagMutation();
  const [deleteTag, { isLoading: isDeletingTag }] = useDeleteCtaTagMutation();
  const isTagMutation = isUpdatingTag || isDeletingTag;
  useCtaBeforeUnload(isBulkMutating || isTagMutation);
  useEffect(() => {
    onMutationBusyChange(isBulkMutating || isTagMutation);
  }, [isBulkMutating, isTagMutation, onMutationBusyChange]);
  useEffect(() => {
    return () => onMutationBusyChange(false);
  }, [onMutationBusyChange]);

  const serverFilter = useMemo<CtaTagFilter>(
    () => ({
      q: debouncedSearch || null,
      tag_type: (tagTypeFilter as CtaTagType) || null,
      tag_set_id: tagSetFilter && tagSetFilter !== '__uncategorized__' ? tagSetFilter : null,
      uncategorized: tagSetFilter === '__uncategorized__',
    }),
    [debouncedSearch, tagSetFilter, tagTypeFilter]
  );
  const queryParams = useMemo(
    () => ({
      q: serverFilter.q ?? undefined,
      tag_type: serverFilter.tag_type ?? undefined,
      tag_set_id: serverFilter.tag_set_id ?? undefined,
      uncategorized: serverFilter.uncategorized || undefined,
      cursor,
      limit: 50,
    }),
    [cursor, serverFilter]
  );
  const { data, isFetching, isError, refetch } = useListCtaTagsQuery(queryParams);
  const {
    data: tagSets,
    isFetching: isTagSetsFetching,
    isError: isTagSetsError,
    refetch: refetchTagSets,
  } = useListCtaTagSetsQuery();
  const {
    currentData: selectedTag,
    isFetching: isSelectedTagFetching,
    isError: isSelectedTagError,
    refetch: refetchSelectedTag,
  } = useGetCtaTagQuery(selectedTagId ?? skipToken);

  const isDataReady = data !== undefined && tagSets !== undefined;
  const isAnyDataFetching = isFetching || isTagSetsFetching;
  const isAnyDataError = isError || isTagSetsError;
  const isInitialLoading = !isDataReady && isAnyDataFetching;
  const isInitialError = !isDataReady && !isAnyDataFetching && isAnyDataError;
  const isRefreshing = isDataReady && isAnyDataFetching;
  const isRefreshError = isDataReady && !isAnyDataFetching && isAnyDataError;
  const isDataInteractionLocked = isAnyDataFetching || isAnyDataError || isBulkMutating || isTagMutation;

  const retryData = useCallback(() => {
    void refetch();
    void refetchTagSets();
  }, [refetch, refetchTagSets]);

  const dataOverlayMessage = isRefreshError
    ? t('cta.ctaDataLoadFailed')
    : isBulkMutating
      ? t('cta.updatingTags')
      : isTagMutation
        ? t('cta.savingCtaChanges')
        : t('cta.refreshingCtaData');

  const resetListState = useCallback(() => {
    setCursor(undefined);
    setCursorStack([]);
    setSelection({ mode: 'ids', ids: new Set() });
    setSelectedTagId(null);
  }, []);

  const togglePageSelection = useCallback(() => {
    const pageIds = data?.items.map((tag) => tag.id) ?? [];
    if (selection.mode === 'ids' && pageIds.length > 0 && pageIds.every((id) => selection.ids.has(id))) {
      setSelection({ mode: 'ids', ids: new Set() });
    } else {
      setSelection({ mode: 'ids', ids: new Set(pageIds) });
    }
  }, [data?.items, selection]);

  const toggleTag = useCallback(
    (id: string) => {
      setSelection((previous) => {
        const ids =
          previous.mode === 'filter' ? new Set(data?.items.map((tag) => tag.id) ?? []) : new Set(previous.ids);
        if (ids.has(id)) {
          ids.delete(id);
        } else {
          ids.add(id);
        }
        return { mode: 'ids', ids };
      });
    },
    [data?.items]
  );

  const runBulk = useCallback(
    async (operation: CtaBulkOperation) => {
      const bulkSelection: S['CtaBulkTagRequest']['selection'] =
        selection.mode === 'ids'
          ? { mode: 'ids', ids: [...selection.ids] }
          : { mode: 'filter', filter: selection.filter };
      try {
        const result = await bulkMutate({ selection: bulkSelection, operation }).unwrap();
        setSelection({ mode: 'ids', ids: new Set() });
        setSelectedTagId(null);
        toast({ title: t('cta.bulkOperationComplete', { count: result.affected_count }), status: 'success' });
      } catch {
        toast({ title: t('cta.bulkOperationFailed'), status: 'error' });
      }
    },
    [bulkMutate, selection, t, toast]
  );

  const hasActiveFilter =
    serverFilter.q !== null ||
    serverFilter.tag_type !== null ||
    serverFilter.tag_set_id !== null ||
    serverFilter.uncategorized;
  const selectedCount = selection.mode === 'ids' ? selection.ids.size : selection.total;
  const pageSelected = Boolean(
    data?.items.length && data.items.every((tag) => selection.mode === 'ids' && selection.ids.has(tag.id))
  );

  return (
    <Flex flexDir="column" gap={3} position="relative" minH="320px">
      <Flex justifyContent="flex-end" position="relative" zIndex={11}>
        <Button size="sm" onClick={onImport} isDisabled={isDataInteractionLocked}>
          {t('cta.importTags')}
        </Button>
      </Flex>
      <HStack position="relative" zIndex={11}>
        <Input
          placeholder={t('common.search')}
          value={search}
          isDisabled={isBulkMutating || isTagMutation}
          onChange={(event) => {
            setSearch(event.target.value);
            resetListState();
          }}
        />
        <Select
          value={tagTypeFilter}
          isDisabled={isDataInteractionLocked}
          onChange={(event) => {
            setTagTypeFilter(event.target.value);
            resetListState();
          }}
          maxW="150px"
        >
          <option value="">{t('common.all')}</option>
          {TAG_TYPES.map((tagType) => (
            <option key={tagType} value={tagType}>
              {tagType}
            </option>
          ))}
        </Select>
        <Select
          value={tagSetFilter}
          isDisabled={isDataInteractionLocked}
          onChange={(event) => {
            setTagSetFilter(event.target.value);
            resetListState();
          }}
          maxW="220px"
        >
          <option value="">{t('common.all')}</option>
          <option value="__uncategorized__">{t('cta.uncategorized')}</option>
          {tagSets?.map((tagSet) => (
            <option key={tagSet.id} value={tagSet.id}>
              {tagSet.name}
            </option>
          ))}
        </Select>
      </HStack>

      <HStack>
        <Checkbox isChecked={pageSelected} isDisabled={isDataInteractionLocked} onChange={togglePageSelection} />
        <Text fontSize="sm">{t('common.selectAll')}</Text>
        {pageSelected && selection.mode === 'ids' && data && data.total_count > data.items.length && (
          <Button
            size="xs"
            isDisabled={isDataInteractionLocked}
            onClick={() => setSelection({ mode: 'filter', filter: serverFilter, total: data.total_count })}
          >
            {t('cta.selectAllMatching', { count: data.total_count })}
          </Button>
        )}
      </HStack>

      {selectedCount > 0 && (
        <HStack flexWrap="wrap">
          <Select
            value={bulkTagSetId}
            isDisabled={isDataInteractionLocked}
            onChange={(event) => setBulkTagSetId(event.target.value)}
            maxW="200px"
          >
            <option value="">{t('cta.selectTagSet')}</option>
            {tagSets?.map((tagSet) => (
              <option key={tagSet.id} value={tagSet.id}>
                {tagSet.name}
              </option>
            ))}
          </Select>
          <Button
            size="sm"
            isDisabled={!bulkTagSetId || isDataInteractionLocked}
            onClick={() => void runBulk({ type: 'add_to_set', tag_set_id: bulkTagSetId })}
          >
            {t('cta.addToSet')}
          </Button>
          <Button
            size="sm"
            isDisabled={!bulkTagSetId || isDataInteractionLocked}
            onClick={() => void runBulk({ type: 'remove_from_set', tag_set_id: bulkTagSetId })}
          >
            {t('cta.removeFromSet')}
          </Button>
          <Select
            value={bulkTagType}
            isDisabled={isDataInteractionLocked}
            onChange={(event) => setBulkTagType(event.target.value as CtaTagType)}
            maxW="150px"
          >
            {TAG_TYPES.map((tagType) => (
              <option key={tagType} value={tagType}>
                {tagType}
              </option>
            ))}
          </Select>
          <Button
            size="sm"
            isDisabled={isDataInteractionLocked}
            onClick={() => void runBulk({ type: 'set_type', tag_type: bulkTagType })}
          >
            {t('cta.setType')}
          </Button>
          <Button
            size="sm"
            colorScheme="red"
            isDisabled={isDataInteractionLocked}
            onClick={() => void runBulk({ type: 'delete' })}
          >
            {t('common.delete')} ({selectedCount})
          </Button>
        </HStack>
      )}

      {data?.items.length === 0 && (
        <VStack py={8}>
          <Text>{hasActiveFilter ? t('cta.noMatchingTags') : t('cta.noTagsImported')}</Text>
          {!hasActiveFilter && (
            <Button size="sm" onClick={onImport} isDisabled={isDataInteractionLocked}>
              {t('cta.importTags')}
            </Button>
          )}
        </VStack>
      )}
      <VStack align="stretch" spacing={1} maxH="360px" overflowY="auto">
        {data?.items.map((tag: CtaTagDTO) => (
          <HStack
            key={tag.id}
            p={2}
            bg={selectedTagId === tag.id ? 'gray.600' : 'transparent'}
            _hover={{ bg: 'gray.600' }}
            cursor="pointer"
            borderRadius="md"
            onClick={() => setSelectedTagId(tag.id)}
          >
            <Checkbox
              isChecked={selection.mode === 'filter' || selection.ids.has(tag.id)}
              isDisabled={isDataInteractionLocked}
              onChange={() => toggleTag(tag.id)}
              onClick={(event) => event.stopPropagation()}
            />
            <Text fontSize="sm" color={TAG_TYPE_COLORS[tag.tag_type] ?? 'gray.500'} flex={1}>
              {tag.canonical_content}
            </Text>
            <Text fontSize="xs">{tag.tag_type}</Text>
            <Text fontSize="xs">{tag.popularity?.toLocaleString() ?? '—'}</Text>
          </HStack>
        ))}
      </VStack>

      <HStack justifyContent="space-between">
        <Text fontSize="sm">{t('cta.totalTags', { count: data?.total_count ?? 0 })}</Text>
        <HStack>
          <Button
            size="sm"
            isDisabled={cursorStack.length === 0 || isDataInteractionLocked}
            onClick={() => {
              const previous = cursorStack.at(-1);
              setCursorStack((stack) => stack.slice(0, -1));
              setCursor(previous);
            }}
          >
            {t('common.back')}
          </Button>
          <Button
            size="sm"
            isDisabled={!data?.next_cursor || isDataInteractionLocked}
            onClick={() => {
              setCursorStack((stack) => [...stack, cursor]);
              setCursor(data?.next_cursor ?? undefined);
            }}
          >
            {t('common.next')}
          </Button>
        </HStack>
      </HStack>

      {selectedTagId && !selectedTag && (
        <Flex position="relative" minH="96px">
          <CtaManagerOverlay
            isOpen
            message={
              isSelectedTagError && !isSelectedTagFetching ? t('cta.tagDetailsLoadFailed') : t('cta.loadingTagDetails')
            }
            onRetry={isSelectedTagError && !isSelectedTagFetching ? () => void refetchSelectedTag() : undefined}
          />
        </Flex>
      )}
      {selectedTag && (
        <TagEditor
          key={selectedTag.id}
          tag={selectedTag}
          tagSets={tagSets ?? []}
          isSaving={isUpdatingTag || isDeletingTag}
          onClose={() => setSelectedTagId(null)}
          onSave={async (body) => {
            try {
              const result = await updateTag({ id: selectedTag.id, body }).unwrap();
              setSelectedTagId(null);
              toast({ title: result.merged ? t('cta.tagsMerged') : t('cta.tagSaved'), status: 'success' });
            } catch {
              toast({ title: t('cta.tagUpdateFailed'), status: 'error' });
            }
          }}
          onDelete={async () => {
            try {
              await deleteTag(selectedTag.id).unwrap();
              setSelectedTagId(null);
              setSelection({ mode: 'ids', ids: new Set() });
              toast({ title: t('cta.tagDeleted'), status: 'success' });
            } catch {
              toast({ title: t('cta.tagDeleteFailed'), status: 'error' });
            }
          }}
        />
      )}
      <CtaManagerOverlay
        isOpen={isRefreshing || isRefreshError || isBulkMutating || isTagMutation}
        message={dataOverlayMessage}
        onRetry={isRefreshError ? retryData : undefined}
      />
      <CtaManagerOverlay
        isOpen={isInitialLoading || isInitialError}
        message={isInitialError ? t('cta.ctaDataLoadFailed') : t('cta.loadingCtaData')}
        onRetry={isInitialError ? retryData : undefined}
        zIndex={20}
      />
      <CtaMutationProgressModal isOpen={isBulkMutating} title={t('cta.updatingTags')} />
    </Flex>
  );
});

TagsTab.displayName = 'TagsTab';

type TagEditorProps = {
  tag: S['CtaTagDetailDTO'];
  tagSets: S['CtaTagSetDTO'][];
  isSaving: boolean;
  onClose: () => void;
  onSave: (body: S['CtaTagUpdate']) => Promise<void>;
  onDelete: () => Promise<void>;
};

const TagEditor = ({ tag, tagSets, isSaving, onClose, onSave, onDelete }: TagEditorProps) => {
  const { t } = useTranslation();
  const [content, setContent] = useState(tag.canonical_content);
  const [tagType, setTagType] = useState<CtaTagType>(tag.tag_type);
  const [popularity, setPopularity] = useState(tag.popularity?.toString() ?? '');
  const [memberships, setMemberships] = useState(new Set(tag.tag_set_ids));

  return (
    <Flex flexDir="column" gap={2} borderWidth={1} borderRadius="md" p={3}>
      <Input value={content} isDisabled={isSaving} onChange={(event) => setContent(event.target.value)} />
      <HStack>
        <Select
          value={tagType}
          isDisabled={isSaving}
          onChange={(event) => setTagType(event.target.value as CtaTagType)}
        >
          {TAG_TYPES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </Select>
        <Input
          type="number"
          min={0}
          value={popularity}
          placeholder={t('cta.unknownPopularity')}
          isDisabled={isSaving}
          onChange={(event) => setPopularity(event.target.value)}
        />
      </HStack>
      <HStack flexWrap="wrap">
        {tagSets.map((tagSet) => (
          <Checkbox
            key={tagSet.id}
            isChecked={memberships.has(tagSet.id)}
            isDisabled={isSaving}
            onChange={() => {
              const next = new Set(memberships);
              if (next.has(tagSet.id)) {
                next.delete(tagSet.id);
              } else {
                next.add(tagSet.id);
              }
              setMemberships(next);
            }}
          >
            {tagSet.name}
          </Checkbox>
        ))}
      </HStack>
      <HStack justifyContent="flex-end">
        <Button size="sm" colorScheme="red" isDisabled={isSaving} onClick={() => void onDelete()}>
          {t('common.delete')}
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isSaving} onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button
          size="sm"
          isLoading={isSaving}
          isDisabled={isSaving || !content.trim()}
          onClick={() =>
            void onSave({
              canonical_content: content,
              tag_type: tagType,
              popularity: popularity === '' ? null : Number(popularity),
              tag_set_ids: [...memberships],
            })
          }
        >
          {t('common.save')}
        </Button>
      </HStack>
    </Flex>
  );
};

import {
  Button,
  ConfirmationAlertDialog,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  IconButton,
  Input,
  Text,
  useToast,
  VStack,
} from '@invoke-ai/ui-library';
import { useCtaBeforeUnload } from 'features/clipTagAutocomplete/hooks/useCtaBeforeUnload';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCreateCtaTagSetMutation,
  useDeleteCtaTagSetMutation,
  useListCtaTagSetsQuery,
  useUpdateCtaTagSetMutation,
} from 'services/api/endpoints/clipTagAutocomplete';

import { CtaManagerOverlay } from './CtaManagerOverlay';

type Props = {
  onViewTags: (tagSetId: string) => void;
  onMutationBusyChange: (isBusy: boolean) => void;
};

export const TagSetsTab = ({ onViewTags, onMutationBusyChange }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const { data: tagSets, isLoading, isFetching, isError, refetch } = useListCtaTagSetsQuery();
  const [createTagSet, { isLoading: isCreating }] = useCreateCtaTagSetMutation();
  const [updateTagSet, { isLoading: isUpdating }] = useUpdateCtaTagSetMutation();
  const [deleteTagSet, { isLoading: isDeleting }] = useDeleteCtaTagSetMutation();
  const isMutating = isCreating || isUpdating || isDeleting;
  const isInteractionLocked = isLoading || isFetching || isMutating || isError;
  const showRetry = isError && !isFetching && !isMutating;
  useCtaBeforeUnload(isMutating);
  useEffect(() => {
    onMutationBusyChange(isMutating);
  }, [isMutating, onMutationBusyChange]);
  useEffect(() => {
    return () => onMutationBusyChange(false);
  }, [onMutationBusyChange]);

  const [newName, setNewName] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const handleCreate = useCallback(async () => {
    if (isMutating || !newName.trim()) {
      return;
    }
    try {
      await createTagSet({ name: newName.trim() }).unwrap();
      setNewName('');
      toast({ title: t('toast.tagSetCreated'), status: 'success' });
    } catch {
      toast({ title: t('toast.tagSetCreateFailed'), status: 'error' });
    }
  }, [createTagSet, isMutating, newName, t, toast]);

  const handleRename = useCallback(
    async (id: string) => {
      if (isMutating || !editingName.trim()) {
        return;
      }
      try {
        await updateTagSet({ id, name: editingName.trim() }).unwrap();
        setEditingId(null);
        toast({ title: t('toast.tagSetUpdated'), status: 'success' });
      } catch {
        toast({ title: t('toast.tagSetUpdateFailed'), status: 'error' });
      }
    },
    [editingName, isMutating, updateTagSet, t, toast]
  );

  const handleDelete = useCallback(async () => {
    if (!pendingDeleteId || isMutating) {
      return;
    }
    const tagSetId = pendingDeleteId;
    setPendingDeleteId(null);
    try {
      await deleteTagSet(tagSetId).unwrap();
      toast({ title: t('toast.tagSetDeleted'), status: 'success' });
    } catch {
      toast({ title: t('toast.tagSetDeleteFailed'), status: 'error' });
    }
  }, [deleteTagSet, isMutating, pendingDeleteId, t, toast]);

  const overlayMessage = showRetry
    ? t('cta.ctaDataLoadFailed')
    : isMutating
      ? t('cta.savingCtaChanges')
      : tagSets === undefined
        ? t('cta.loadingCtaData')
        : t('cta.refreshingCtaData');

  return (
    <Flex flexDir="column" gap={4} position="relative" minH="240px">
      <HStack>
        <FormControl isDisabled={isInteractionLocked}>
          <FormLabel>{t('cta.newTagSetName')}</FormLabel>
          <Input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder={t('cta.tagNamePlaceholder')}
          />
        </FormControl>
        <Button onClick={handleCreate} isLoading={isCreating} isDisabled={isInteractionLocked || !newName.trim()}>
          {t('common.create')}
        </Button>
      </HStack>

      <VStack align="stretch" spacing={2}>
        {tagSets?.map((set) => (
          <HStack key={set.id} p={2} bg="gray.700" borderRadius="md">
            {editingId === set.id ? (
              <>
                <Input
                  value={editingName}
                  isDisabled={isInteractionLocked}
                  onChange={(e) => setEditingName(e.target.value)}
                  size="sm"
                />
                <Button
                  size="sm"
                  isLoading={isUpdating}
                  isDisabled={isInteractionLocked}
                  onClick={() => handleRename(set.id)}
                >
                  {t('common.save')}
                </Button>
                <Button size="sm" variant="ghost" isDisabled={isInteractionLocked} onClick={() => setEditingId(null)}>
                  {t('common.cancel')}
                </Button>
              </>
            ) : (
              <>
                <Text flex={1}>{set.name}</Text>
                <Text fontSize="sm" color="gray.400">
                  {set.tag_count} {t('common.tags')}
                </Text>
                <Text fontSize="sm" color="gray.400">
                  {t('cta.modelCount', { count: set.model_count })}
                </Text>
                <Button size="sm" variant="ghost" isDisabled={isInteractionLocked} onClick={() => onViewTags(set.id)}>
                  {t('cta.viewTags')}
                </Button>
                <IconButton
                  aria-label={t('common.rename')}
                  icon={<PencilIcon />}
                  size="sm"
                  variant="ghost"
                  isDisabled={isInteractionLocked}
                  onClick={() => {
                    setEditingId(set.id);
                    setEditingName(set.name);
                  }}
                />
                <IconButton
                  aria-label={t('common.delete')}
                  icon={<TrashIcon />}
                  size="sm"
                  variant="ghost"
                  colorScheme="red"
                  isDisabled={isInteractionLocked}
                  onClick={() => setPendingDeleteId(set.id)}
                />
              </>
            )}
          </HStack>
        ))}
      </VStack>
      <CtaManagerOverlay
        isOpen={isLoading || isFetching || isMutating || isError}
        message={overlayMessage}
        onRetry={showRetry ? () => void refetch() : undefined}
      />
      <ConfirmationAlertDialog
        title={t('cta.deleteTagSet')}
        isOpen={pendingDeleteId !== null}
        onClose={() => setPendingDeleteId(null)}
        cancelButtonText={t('common.cancel')}
        acceptButtonText={t('common.delete')}
        acceptCallback={() => void handleDelete()}
        cancelCallback={() => setPendingDeleteId(null)}
        useInert={false}
      >
        <VStack align="stretch">
          <Text>{t('cta.deleteTagSetModelConfigWarning')}</Text>
          <Text>{t('cta.deleteTagSetPreserveTags')}</Text>
        </VStack>
      </ConfirmationAlertDialog>
    </Flex>
  );
};

const PencilIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    width={16}
    height={16}
  >
    <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
    <path d="m15 5 4 4" />
  </svg>
);

const TrashIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    width={16}
    height={16}
  >
    <path d="M3 6h18" />
    <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
  </svg>
);

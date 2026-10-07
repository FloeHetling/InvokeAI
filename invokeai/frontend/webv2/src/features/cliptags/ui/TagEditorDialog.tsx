import type { ClipTag, ClipTagDetail, ClipTagSet, ClipTagType, ClipTagUpdate } from '@features/cliptags/core/types';

import { Checkbox, Dialog, HStack, Input, Portal, Spinner, Stack, Text } from '@chakra-ui/react';
import { deleteClipTag, updateClipTag } from '@features/cliptags/data/api';
import { clipTagOptions } from '@features/cliptags/data/queries';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { EmptyState } from '@platform/ui/EmptyState';
import { Field } from '@platform/ui/Field';
import { Scrollable } from '@platform/ui/Scrollable';
import { useQuery } from '@tanstack/react-query';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import { OptionSelect } from './OptionSelect';
import { buildTagUpdate, getTagFormValues, parsePopularity, type TagFormValues } from './tagEditing';
import { isClipTagType, useTagTypeOptions } from './tagTypes';

export interface TagEditorDialogProps {
  tagSets: readonly ClipTagSet[];
  /** The tag being edited, or null while the dialog is closed. */
  tag: ClipTag | null;
  onClose: () => void;
}

/** Edits one tag: its text, type, popularity and tag set membership, or deletes it. */
export const TagEditorDialog = ({ onClose, tag, tagSets }: TagEditorDialogProps) => {
  const { t } = useTranslation();
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);
  const update = useClipTagWrite(({ id, changes }: { changes: ClipTagUpdate; id: string }) =>
    updateClipTag(id, changes)
  );
  const remove = useClipTagWrite((id: string) => deleteClipTag(id));
  const isPending = update.isPending || remove.isPending;
  // The form keeps its tag while the dialog animates out.
  const shown = useExitRetainedValue(tag);
  const { release } = shown;

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open && !isPending) {
        onClose();
      }
    },
    [isPending, onClose]
  );
  const handleSave = useCallback(
    async (changes: ClipTagUpdate) => {
      if (!tag) {
        return;
      }

      try {
        const result = await update.mutateAsync({ changes, id: tag.id });

        if (result.merged) {
          notify('success', t('cliptags.manager.editor.merged'), t('cliptags.manager.editor.mergedDescription'));
        } else {
          notify('success', t('cliptags.manager.editor.saved'));
        }

        onClose();
      } catch (error) {
        notify('error', t('cliptags.manager.editor.saveFailed'), getApiErrorMessage(error, ''));
      }
    },
    [onClose, t, tag, update]
  );
  const handleDelete = useCallback(async () => {
    if (!tag) {
      return;
    }

    try {
      await remove.mutateAsync(tag.id);
      notify('success', t('cliptags.manager.editor.deleted'));
      onClose();
    } catch (error) {
      notify('error', t('cliptags.manager.editor.deleteFailed'), getApiErrorMessage(error, ''));
    }
  }, [onClose, remove, t, tag]);
  const openDeleteConfirm = useCallback(() => setIsConfirmingDelete(true), []);
  const closeDeleteConfirm = useCallback(() => setIsConfirmingDelete(false), []);

  return (
    <>
      <Dialog.Root
        closeOnEscape={!isPending}
        closeOnInteractOutside={!isPending}
        lazyMount
        open={tag !== null}
        scrollBehavior="inside"
        size="sm"
        unmountOnExit
        onExitComplete={release}
        onOpenChange={handleOpenChange}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
                <Dialog.Title>{t('cliptags.manager.editor.title')}</Dialog.Title>
              </Dialog.Header>
              {shown.value ? (
                <TagEditorBody
                  key={shown.generation}
                  isPending={isPending}
                  isSaving={update.isPending}
                  tag={shown.value}
                  tagSets={tagSets}
                  onClose={onClose}
                  onDelete={openDeleteConfirm}
                  onSave={handleSave}
                />
              ) : null}
              <Dialog.CloseTrigger asChild>
                <CloseButton disabled={isPending} />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      <ConfirmDialog
        body={t('cliptags.manager.editor.deleteBody', { name: tag?.content ?? '' })}
        confirmLabel={t('common.delete')}
        isOpen={isConfirmingDelete}
        title={t('cliptags.manager.editor.deleteTitle')}
        onClose={closeDeleteConfirm}
        onConfirm={handleDelete}
      />
    </>
  );
};

const TagEditorBody = ({
  isPending,
  isSaving,
  onClose,
  onDelete,
  onSave,
  tag,
  tagSets,
}: {
  isPending: boolean;
  isSaving: boolean;
  tag: ClipTag;
  tagSets: readonly ClipTagSet[];
  onClose: () => void;
  onDelete: () => void;
  onSave: (changes: ClipTagUpdate) => void;
}) => {
  const { t } = useTranslation();
  const detail = useQuery(clipTagOptions(tag.id));
  const retry = useCallback(() => void detail.refetch(), [detail]);

  if (detail.isPending) {
    return (
      <Dialog.Body>
        <HStack color="fg.muted" gap="2" justify="center" minH="24" role="status">
          <Spinner />
          <Text fontSize="md">{t('cliptags.manager.editor.loading')}</Text>
        </HStack>
      </Dialog.Body>
    );
  }

  if (detail.isError) {
    return (
      <Dialog.Body>
        <EmptyState
          danger
          description={getApiErrorMessage(detail.error, '')}
          title={t('cliptags.manager.editor.loadFailed')}
        >
          <Button variant="outline" onClick={retry}>
            {t('common.retry')}
          </Button>
        </EmptyState>
      </Dialog.Body>
    );
  }

  return (
    <TagForm
      detail={detail.data}
      isPending={isPending}
      isSaving={isSaving}
      tagSets={tagSets}
      onClose={onClose}
      onDelete={onDelete}
      onSave={onSave}
    />
  );
};

const TagForm = ({
  detail,
  isPending,
  isSaving,
  onClose,
  onDelete,
  onSave,
  tagSets,
}: {
  detail: ClipTagDetail;
  isPending: boolean;
  isSaving: boolean;
  tagSets: readonly ClipTagSet[];
  onClose: () => void;
  onDelete: () => void;
  onSave: (changes: ClipTagUpdate) => void;
}) => {
  const { t } = useTranslation();
  const typeOptions = useTagTypeOptions();
  const [values, setValues] = useState<TagFormValues>(() => getTagFormValues(detail));
  const isContentValid = values.content.trim() !== '';
  const isPopularityValid = parsePopularity(values.popularity) !== null;

  const handleContentChange = useCallback((event: { currentTarget: { value: string } }) => {
    const content = event.currentTarget.value;
    setValues((current) => ({ ...current, content }));
  }, []);
  const handleTypeChange = useCallback((value: string) => {
    if (isClipTagType(value)) {
      const type: ClipTagType = value;
      setValues((current) => ({ ...current, type }));
    }
  }, []);
  const handlePopularityChange = useCallback((event: { currentTarget: { value: string } }) => {
    const popularity = event.currentTarget.value;
    setValues((current) => ({ ...current, popularity }));
  }, []);
  const toggleTagSet = useCallback((tagSetId: string) => {
    setValues((current) => {
      const tagSetIds = new Set(current.tagSetIds);

      if (!tagSetIds.delete(tagSetId)) {
        tagSetIds.add(tagSetId);
      }

      return { ...current, tagSetIds };
    });
  }, []);
  const handleSubmit = useCallback(
    (event: { preventDefault: () => void }) => {
      event.preventDefault();

      if (!isContentValid || !isPopularityValid) {
        return;
      }

      const changes = buildTagUpdate(detail, values);

      if (changes) {
        onSave(changes);
      } else {
        onClose();
      }
    },
    [detail, isContentValid, isPopularityValid, onClose, onSave, values]
  );

  return (
    <form onSubmit={handleSubmit}>
      <Dialog.Body>
        <Stack gap="4">
          <Field
            disabled={isPending}
            error={isContentValid ? null : t('cliptags.manager.editor.contentRequired')}
            label={t('cliptags.manager.editor.content')}
          >
            <Input autoComplete="off" value={values.content} onChange={handleContentChange} />
          </Field>
          <HStack align="flex-start" flexWrap="wrap" gap="3">
            <Field disabled={isPending} flex="1 1 9rem" label={t('cliptags.manager.editor.type')}>
              <OptionSelect
                inDialog
                label={t('cliptags.manager.editor.type')}
                options={typeOptions}
                value={values.type}
                onChange={handleTypeChange}
              />
            </Field>
            <Field
              disabled={isPending}
              error={isPopularityValid ? null : t('cliptags.manager.editor.popularityInvalid')}
              flex="1 1 9rem"
              label={t('cliptags.manager.editor.popularity')}
            >
              <Input
                autoComplete="off"
                inputMode="numeric"
                placeholder={t('cliptags.manager.editor.popularityPlaceholder')}
                value={values.popularity}
                onChange={handlePopularityChange}
              />
            </Field>
          </HStack>
          <TagSetMembership
            disabled={isPending}
            selected={values.tagSetIds}
            tagSets={tagSets}
            onToggle={toggleTagSet}
          />
        </Stack>
      </Dialog.Body>
      <Dialog.Footer>
        <Button colorPalette="red" disabled={isPending} me="auto" variant="ghost" onClick={onDelete}>
          {t('common.delete')}
        </Button>
        <Button disabled={isPending} variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button
          disabled={isPending || !isContentValid || !isPopularityValid}
          loading={isSaving}
          type="submit"
          variant="solid"
        >
          {t('common.save')}
        </Button>
      </Dialog.Footer>
    </form>
  );
};

const TagSetMembership = ({
  disabled,
  onToggle,
  selected,
  tagSets,
}: {
  disabled: boolean;
  selected: ReadonlySet<string>;
  tagSets: readonly ClipTagSet[];
  onToggle: (tagSetId: string) => void;
}) => {
  const { t } = useTranslation();

  return (
    <Field disabled={disabled} label={t('cliptags.manager.editor.tagSets')}>
      {tagSets.length === 0 ? (
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.manager.editor.noTagSets')}
        </Text>
      ) : (
        <Scrollable
          borderColor="border.subtle"
          borderWidth="1px"
          label={t('cliptags.manager.editor.tagSets')}
          maxH="44"
          rounded="md"
        >
          <Stack gap="1" p="2">
            {tagSets.map((tagSet) => (
              <TagSetCheckbox
                key={tagSet.id}
                checked={selected.has(tagSet.id)}
                disabled={disabled}
                tagSet={tagSet}
                onToggle={onToggle}
              />
            ))}
          </Stack>
        </Scrollable>
      )}
    </Field>
  );
};

const TagSetCheckbox = ({
  checked,
  disabled,
  onToggle,
  tagSet,
}: {
  checked: boolean;
  disabled: boolean;
  tagSet: ClipTagSet;
  onToggle: (tagSetId: string) => void;
}) => {
  const handleCheckedChange = useCallback(() => onToggle(tagSet.id), [onToggle, tagSet.id]);

  return (
    <Checkbox.Root
      checked={checked}
      colorPalette="accent"
      disabled={disabled}
      size="sm"
      onCheckedChange={handleCheckedChange}
    >
      <Checkbox.HiddenInput />
      <Checkbox.Control />
      <Checkbox.Label minW="0" overflowWrap="anywhere">
        {tagSet.name}
      </Checkbox.Label>
    </Checkbox.Root>
  );
};

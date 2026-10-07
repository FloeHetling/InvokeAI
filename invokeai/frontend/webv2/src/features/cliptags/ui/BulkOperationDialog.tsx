import type {
  ClipTagBulkOperation,
  ClipTagBulkResult,
  ClipTagBulkSelection,
  ClipTagSet,
  ClipTagType,
} from '@features/cliptags/core/types';
import type { TFunction } from 'i18next';

import { Dialog, Portal, Progress, Stack, Text } from '@chakra-ui/react';
import { bulkMutateClipTags } from '@features/cliptags/data/api';
import { formatCount } from '@platform/i18n/languages';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import { ErrorAlert } from './ErrorAlert';
import { OptionSelect, type Option } from './OptionSelect';
import { isClipTagType, useTagTypeOptions } from './tagTypes';

export type BulkOperationKind = ClipTagBulkOperation['type'];

export interface BulkOperationDialogProps {
  /** The operation to confirm, or null while the dialog is closed. */
  kind: BulkOperationKind | null;
  /** How many tags the operation will touch. */
  count: number;
  /** Resolved when the operation is applied, so it reflects the selection at that moment. */
  getSelection: () => ClipTagBulkSelection;
  tagSets: readonly ClipTagSet[];
  onClose: () => void;
  /** After the operation succeeded and the dialog asked to close. */
  onDone: () => void;
}

/**
 * Confirms one bulk operation on the selected tags and runs it. A bulk operation over thousands of tags can take a
 * while, so the dialog stays open and inert, showing progress, until the server answers; a failure is shown in
 * place so the operation can be retried.
 */
export const BulkOperationDialog = ({
  count,
  getSelection,
  kind,
  onClose,
  onDone,
  tagSets,
}: BulkOperationDialogProps) => {
  const { t } = useTranslation();
  const bulk = useClipTagWrite(
    ({ operation, selection: target }: { operation: ClipTagBulkOperation; selection: ClipTagBulkSelection }) =>
      bulkMutateClipTags(target, operation)
  );
  const { isPending, reset } = bulk;
  // The form keeps its operation while the dialog animates out.
  const shown = useExitRetainedValue(kind);
  const { release } = shown;

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open && !isPending) {
        onClose();
      }
    },
    [isPending, onClose]
  );
  const handleExitComplete = useCallback(() => {
    release();
    reset();
  }, [release, reset]);
  const handleApply = useCallback(
    async (operation: ClipTagBulkOperation) => {
      try {
        const result = await bulk.mutateAsync({ operation, selection: getSelection() });

        notifyBulkResult(operation.type, result, count, t);
        onDone();
      } catch {
        // The dialog shows the failure from the mutation's state.
      }
    },
    [bulk, count, getSelection, onDone, t]
  );

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      lazyMount
      open={kind !== null}
      role={shown.value === 'delete' ? 'alertdialog' : 'dialog'}
      scrollBehavior="inside"
      size="sm"
      unmountOnExit
      onExitComplete={handleExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            {shown.value ? (
              <BulkOperationForm
                key={shown.generation}
                count={count}
                error={bulk.isError ? getApiErrorMessage(bulk.error, t('cliptags.manager.bulk.failed')) : null}
                isPending={isPending}
                kind={shown.value}
                tagSets={tagSets}
                onApply={handleApply}
                onClose={onClose}
              />
            ) : null}
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isPending} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const notifyBulkResult = (
  kind: BulkOperationKind,
  result: ClipTagBulkResult,
  announced: number,
  t: TFunction
): void => {
  const notes = [
    result.mergedCount > 0
      ? t('cliptags.manager.bulk.merged', { count: result.mergedCount, total: formatCount(result.mergedCount) })
      : '',
    // The dialog confirmed `announced` tags; rows changed elsewhere in the meantime can make the server match others.
    result.selectedCount !== announced
      ? t('cliptags.manager.bulk.countChanged', {
          announced: formatCount(announced),
          count: result.selectedCount,
          total: formatCount(result.selectedCount),
        })
      : '',
  ].filter(Boolean);

  notify(
    'success',
    t(kind === 'delete' ? 'cliptags.manager.bulk.deletedComplete' : 'cliptags.manager.bulk.complete', {
      count: result.affectedCount,
      total: formatCount(result.affectedCount),
    }),
    notes.length > 0 ? notes.join(' ') : undefined
  );
};

const BulkOperationForm = ({
  count,
  error,
  isPending,
  kind,
  onApply,
  onClose,
  tagSets,
}: {
  count: number;
  error: string | null;
  isPending: boolean;
  kind: BulkOperationKind;
  tagSets: readonly ClipTagSet[];
  onApply: (operation: ClipTagBulkOperation) => void;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const typeOptions = useTagTypeOptions();
  const [tagSetId, setTagSetId] = useState('');
  const [tagType, setTagType] = useState<ClipTagType>('general');
  const tagSetOptions = useMemo<Option[]>(
    () => tagSets.map((tagSet) => ({ label: tagSet.name, value: tagSet.id })),
    [tagSets]
  );
  const needsTagSet = kind === 'add_to_set' || kind === 'remove_from_set';
  const canApply = !needsTagSet || tagSetId !== '';

  const handleTypeChange = useCallback((value: string) => {
    if (isClipTagType(value)) {
      setTagType(value);
    }
  }, []);
  const handleSubmit = useCallback(
    (event: { preventDefault: () => void }) => {
      event.preventDefault();

      if (!canApply || isPending) {
        return;
      }

      switch (kind) {
        case 'add_to_set':
        case 'remove_from_set':
          onApply({ tagSetId, type: kind });
          break;
        case 'set_type':
          onApply({ tagType, type: kind });
          break;
        case 'delete':
          onApply({ type: kind });
          break;
      }
    },
    [canApply, isPending, kind, onApply, tagSetId, tagType]
  );

  return (
    <form onSubmit={handleSubmit}>
      <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
        <Stack gap="0.5">
          <Dialog.Title>{t(`cliptags.manager.bulk.titles.${kind}`)}</Dialog.Title>
          <Dialog.Description>
            {t('cliptags.manager.bulk.selected', { count, total: formatCount(count) })}
          </Dialog.Description>
        </Stack>
      </Dialog.Header>
      <Dialog.Body>
        <Stack gap="4">
          {needsTagSet ? (
            tagSets.length === 0 ? (
              <Text color="fg.muted" fontSize="md">
                {t('cliptags.manager.bulk.noTagSets')}
              </Text>
            ) : (
              <Field disabled={isPending} label={t('cliptags.manager.bulk.tagSet')}>
                <OptionSelect
                  inDialog
                  label={t('cliptags.manager.bulk.tagSet')}
                  options={tagSetOptions}
                  placeholder={t('cliptags.manager.bulk.selectTagSet')}
                  value={tagSetId}
                  onChange={setTagSetId}
                />
              </Field>
            )
          ) : null}
          {kind === 'set_type' ? (
            <Field disabled={isPending} label={t('cliptags.manager.bulk.tagType')}>
              <OptionSelect
                inDialog
                label={t('cliptags.manager.bulk.tagType')}
                options={typeOptions}
                value={tagType}
                onChange={handleTypeChange}
              />
            </Field>
          ) : null}
          {kind === 'delete' ? (
            <Text fontSize="md">{t('cliptags.manager.bulk.deleteWarning', { count, total: formatCount(count) })}</Text>
          ) : null}
          {error ? <ErrorAlert message={error} /> : null}
          {isPending ? (
            <Stack gap="1.5" role="status">
              <Text color="fg.muted" fontSize="md">
                {t('cliptags.manager.bulk.working')}
              </Text>
              <Progress.Root aria-label={t('cliptags.manager.bulk.working')} size="md" value={null}>
                <Progress.Track>
                  <Progress.Range />
                </Progress.Track>
              </Progress.Root>
            </Stack>
          ) : null}
        </Stack>
      </Dialog.Body>
      <Dialog.Footer>
        <Button disabled={isPending} variant="ghost" onClick={onClose}>
          {t('common.cancel')}
        </Button>
        <Button
          colorPalette={kind === 'delete' ? 'red' : 'accent'}
          disabled={!canApply || isPending}
          loading={isPending}
          type="submit"
          variant="solid"
        >
          {t(`cliptags.manager.bulk.apply.${kind}`)}
        </Button>
      </Dialog.Footer>
    </form>
  );
};

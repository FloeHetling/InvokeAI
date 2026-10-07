import type { ClipTagSet } from '@features/cliptags/core/types';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { createClipTagSet, deleteClipTagSet, renameClipTagSet } from '@features/cliptags/data/api';
import { clipTagSetsOptions } from '@features/cliptags/data/queries';
import { formatCount } from '@platform/i18n/languages';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { EmptyState } from '@platform/ui/EmptyState';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import { FoldersIcon, PencilIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import { QueryState } from './QueryState';

type NameDialogTarget = { mode: 'create' } | { mode: 'rename'; tagSet: ClipTagSet };

export interface TagSetsTabProps {
  /** Opens the Tags tab filtered to this tag set. */
  onViewTags: (tagSetId: string) => void;
}

/** Create, rename and delete tag sets, and jump to the tags in one. */
export const TagSetsTab = ({ onViewTags }: TagSetsTabProps) => {
  const { t } = useTranslation();
  const query = useQuery(clipTagSetsOptions());
  const [nameTarget, setNameTarget] = useState<NameDialogTarget | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ClipTagSet | null>(null);
  const create = useClipTagWrite((name: string) => createClipTagSet(name));
  const rename = useClipTagWrite(({ id, name }: { id: string; name: string }) => renameClipTagSet(id, name));
  const remove = useClipTagWrite((id: string) => deleteClipTagSet(id));
  const tagSets = query.data;

  const openCreate = useCallback(() => setNameTarget({ mode: 'create' }), []);
  const closeNameDialog = useCallback(() => setNameTarget(null), []);
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  // A rejected submit keeps the name dialog open; the failure is reported here.
  const handleSubmitName = useCallback(
    async (name: string) => {
      const target = nameTarget;

      try {
        if (target?.mode === 'rename') {
          await rename.mutateAsync({ id: target.tagSet.id, name });
          notify('success', t('cliptags.manager.tagSets.renamed'));
        } else {
          await create.mutateAsync(name);
          notify('success', t('cliptags.manager.tagSets.created'));
        }
      } catch (error) {
        notify(
          'error',
          target?.mode === 'rename'
            ? t('cliptags.manager.tagSets.renameFailed')
            : t('cliptags.manager.tagSets.createFailed'),
          getApiErrorMessage(error, '')
        );

        throw error;
      }
    },
    [create, nameTarget, rename, t]
  );
  const handleDelete = useCallback(async () => {
    if (!deleteTarget) {
      return;
    }

    try {
      await remove.mutateAsync(deleteTarget.id);
      notify('success', t('cliptags.manager.tagSets.deleted'));
    } catch (error) {
      notify('error', t('cliptags.manager.tagSets.deleteFailed'), getApiErrorMessage(error, ''));
    }
  }, [deleteTarget, remove, t]);

  const isRename = nameTarget?.mode === 'rename';

  return (
    <Stack flex="1" gap="3" minH="0">
      <HStack gap="3" justify="space-between" minH="8">
        <Text color="fg.muted" fontSize="md">
          {tagSets
            ? t('cliptags.manager.tagSets.count', { count: tagSets.length, total: formatCount(tagSets.length) })
            : ''}
        </Text>
        <Button size="sm" variant="outline" onClick={openCreate}>
          <PlusIcon />
          {t('cliptags.manager.tagSets.new')}
        </Button>
      </HStack>
      <QueryState errorTitle={t('cliptags.manager.tagSets.loadFailed')} query={query}>
        {tagSets && tagSets.length > 0 ? (
          <Scrollable flex="1" label={t('cliptags.manager.tagSets.listLabel')} minH="0">
            <ListStack label={t('cliptags.manager.tagSets.listLabel')}>
              {tagSets.map((tagSet) => (
                <TagSetRow
                  key={tagSet.id}
                  tagSet={tagSet}
                  onDelete={setDeleteTarget}
                  onRename={(target) => setNameTarget({ mode: 'rename', tagSet: target })}
                  onViewTags={onViewTags}
                />
              ))}
            </ListStack>
          </Scrollable>
        ) : (
          <EmptyState
            description={t('cliptags.manager.tagSets.emptyDescription')}
            icon={<Icon as={FoldersIcon} />}
            title={t('cliptags.manager.tagSets.empty')}
          />
        )}
      </QueryState>
      <RenameDialog
        initialName={isRename ? nameTarget.tagSet.name : ''}
        isOpen={nameTarget !== null}
        label={t('cliptags.manager.tagSets.name')}
        submitLabel={isRename ? t('common.rename') : t('cliptags.manager.tagSets.create')}
        title={isRename ? t('cliptags.manager.tagSets.renameTitle') : t('cliptags.manager.tagSets.newTitle')}
        onClose={closeNameDialog}
        onSubmit={handleSubmitName}
      />
      <ConfirmDialog
        body={
          deleteTarget ? (
            <Stack gap="2">
              <Text fontSize="md">{t('cliptags.manager.tagSets.deleteBody', { name: deleteTarget.name })}</Text>
              <Text fontSize="md">{t('cliptags.manager.tagSets.deletePreserveTags')}</Text>
              {deleteTarget.modelCount > 0 ? (
                <Text fontSize="md">
                  {t('cliptags.manager.tagSets.deleteModelWarning', {
                    count: deleteTarget.modelCount,
                    total: formatCount(deleteTarget.modelCount),
                  })}
                </Text>
              ) : null}
            </Stack>
          ) : null
        }
        confirmLabel={t('common.delete')}
        isOpen={deleteTarget !== null}
        title={t('cliptags.manager.tagSets.deleteTitle')}
        onClose={closeDeleteDialog}
        onConfirm={handleDelete}
      />
    </Stack>
  );
};

const TagSetRow = ({
  onDelete,
  onRename,
  onViewTags,
  tagSet,
}: {
  tagSet: ClipTagSet;
  onDelete: (tagSet: ClipTagSet) => void;
  onRename: (tagSet: ClipTagSet) => void;
  onViewTags: (tagSetId: string) => void;
}) => {
  const { t } = useTranslation();
  const renameLabel = t('cliptags.manager.tagSets.renameNamed', { name: tagSet.name });
  const deleteLabel = t('cliptags.manager.tagSets.deleteNamed', { name: tagSet.name });

  return (
    <ListItem
      actions={
        <HStack gap="1">
          <Button
            aria-label={t('cliptags.manager.tagSets.viewTagsNamed', { name: tagSet.name })}
            size="xs"
            variant="ghost"
            onClick={() => onViewTags(tagSet.id)}
          >
            {t('cliptags.manager.tagSets.viewTags')}
          </Button>
          <Tooltip content={renameLabel}>
            <IconButton
              aria-label={renameLabel}
              color="fg.muted"
              size="sm"
              variant="ghost"
              onClick={() => onRename(tagSet)}
            >
              <PencilIcon />
            </IconButton>
          </Tooltip>
          <Tooltip content={deleteLabel}>
            <IconButton
              aria-label={deleteLabel}
              color="fg.muted"
              size="sm"
              variant="ghost"
              onClick={() => onDelete(tagSet)}
            >
              <Trash2Icon />
            </IconButton>
          </Tooltip>
        </HStack>
      }
      density="comfortable"
      description={t('cliptags.manager.tagSets.summary', {
        models: t('cliptags.manager.tagSets.modelCount', {
          count: tagSet.modelCount,
          total: formatCount(tagSet.modelCount),
        }),
        tags: t('cliptags.manager.tagSets.tagCount', { count: tagSet.tagCount, total: formatCount(tagSet.tagCount) }),
      })}
      title={tagSet.name}
      titleTruncate="end"
    />
  );
};

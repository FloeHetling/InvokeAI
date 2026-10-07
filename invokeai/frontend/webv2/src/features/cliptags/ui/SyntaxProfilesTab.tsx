import type { ClipSyntaxProfile } from '@features/cliptags/core/types';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { deleteClipSyntaxProfile } from '@features/cliptags/data/api';
import { clipSyntaxProfilesOptions } from '@features/cliptags/data/queries';
import { formatCount } from '@platform/i18n/languages';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { EmptyState } from '@platform/ui/EmptyState';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import { PencilIcon, PlusIcon, SlidersHorizontalIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import { QueryState } from './QueryState';
import { SYNTAX_PROFILE_OPTIONS, SyntaxProfileDialog, type SyntaxProfileDialogTarget } from './SyntaxProfileDialog';

/** Create, edit and delete the syntax profiles that decide how a tag is written into a prompt. */
export const SyntaxProfilesTab = () => {
  const { t } = useTranslation();
  const query = useQuery(clipSyntaxProfilesOptions());
  const [dialogTarget, setDialogTarget] = useState<SyntaxProfileDialogTarget | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ClipSyntaxProfile | null>(null);
  const remove = useClipTagWrite((id: string) => deleteClipSyntaxProfile(id));
  const profiles = query.data;

  const openCreate = useCallback(() => setDialogTarget({ mode: 'create' }), []);
  const closeDialog = useCallback(() => setDialogTarget(null), []);
  const closeDeleteDialog = useCallback(() => setDeleteTarget(null), []);
  const handleDelete = useCallback(async () => {
    if (!deleteTarget) {
      return;
    }

    try {
      await remove.mutateAsync(deleteTarget.id);
      notify('success', t('cliptags.manager.syntaxProfiles.deleted'));
    } catch (error) {
      notify('error', t('cliptags.manager.syntaxProfiles.deleteFailed'), getApiErrorMessage(error, ''));
    }
  }, [deleteTarget, remove, t]);

  return (
    <Stack flex="1" gap="3" minH="0">
      <HStack gap="3" justify="space-between" minH="8">
        <Text color="fg.muted" fontSize="md">
          {profiles
            ? t('cliptags.manager.syntaxProfiles.count', {
                count: profiles.length,
                total: formatCount(profiles.length),
              })
            : ''}
        </Text>
        <Button size="sm" variant="outline" onClick={openCreate}>
          <PlusIcon />
          {t('cliptags.manager.syntaxProfiles.new')}
        </Button>
      </HStack>
      <QueryState errorTitle={t('cliptags.manager.syntaxProfiles.loadFailed')} query={query}>
        {profiles && profiles.length > 0 ? (
          <Scrollable flex="1" label={t('cliptags.manager.syntaxProfiles.listLabel')} minH="0">
            <ListStack label={t('cliptags.manager.syntaxProfiles.listLabel')}>
              {profiles.map((profile) => (
                <SyntaxProfileRow
                  key={profile.id}
                  profile={profile}
                  onDelete={setDeleteTarget}
                  onEdit={(target) => setDialogTarget({ mode: 'edit', profile: target })}
                />
              ))}
            </ListStack>
          </Scrollable>
        ) : (
          <EmptyState
            description={t('cliptags.manager.syntaxProfiles.emptyDescription')}
            icon={<Icon as={SlidersHorizontalIcon} />}
            title={t('cliptags.manager.syntaxProfiles.empty')}
          />
        )}
      </QueryState>
      <SyntaxProfileDialog target={dialogTarget} onClose={closeDialog} />
      <ConfirmDialog
        body={
          deleteTarget ? (
            <Stack gap="2">
              <Text fontSize="md">{t('cliptags.manager.syntaxProfiles.deleteBody', { name: deleteTarget.name })}</Text>
              <Text fontSize="md">{t('cliptags.manager.syntaxProfiles.deleteWarning')}</Text>
            </Stack>
          ) : null
        }
        confirmLabel={t('common.delete')}
        isOpen={deleteTarget !== null}
        title={t('cliptags.manager.syntaxProfiles.deleteTitle')}
        onClose={closeDeleteDialog}
        onConfirm={handleDelete}
      />
    </Stack>
  );
};

const SyntaxProfileRow = ({
  onDelete,
  onEdit,
  profile,
}: {
  profile: ClipSyntaxProfile;
  onDelete: (profile: ClipSyntaxProfile) => void;
  onEdit: (profile: ClipSyntaxProfile) => void;
}) => {
  const { t } = useTranslation();
  const editLabel = t('cliptags.manager.syntaxProfiles.editNamed', { name: profile.name });
  const deleteLabel = t('cliptags.manager.syntaxProfiles.deleteNamed', { name: profile.name });
  const enabled = SYNTAX_PROFILE_OPTIONS.filter(({ option }) => profile[option]).length;

  return (
    <ListItem
      actions={
        <HStack gap="1">
          <Tooltip content={editLabel}>
            <IconButton
              aria-label={editLabel}
              color="fg.muted"
              size="sm"
              variant="ghost"
              onClick={() => onEdit(profile)}
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
              onClick={() => onDelete(profile)}
            >
              <Trash2Icon />
            </IconButton>
          </Tooltip>
        </HStack>
      }
      density="comfortable"
      description={t('cliptags.manager.syntaxProfiles.optionsEnabled', {
        enabled,
        total: SYNTAX_PROFILE_OPTIONS.length,
      })}
      title={profile.name}
      titleTruncate="end"
    />
  );
};

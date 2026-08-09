import {
  Button,
  ConfirmationAlertDialog,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  IconButton,
  Input,
  SimpleGrid,
  Switch,
  Text,
  useToast,
  VStack,
} from '@invoke-ai/ui-library';
import { useCtaBeforeUnload } from 'features/clipTagAutocomplete/hooks/useCtaBeforeUnload';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCreateCtaSyntaxProfileMutation,
  useDeleteCtaSyntaxProfileMutation,
  useListCtaSyntaxProfilesQuery,
  useUpdateCtaSyntaxProfileMutation,
} from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';

import { CtaManagerOverlay } from './CtaManagerOverlay';

type Props = {
  onMutationBusyChange: (isBusy: boolean) => void;
};

type CtaSyntaxProfileDTO = S['CtaSyntaxProfileDTO'];

export const SyntaxProfilesTab = ({ onMutationBusyChange }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const { data: profiles, isLoading, isFetching, isError, refetch } = useListCtaSyntaxProfilesQuery();
  const [createProfile, { isLoading: isCreating }] = useCreateCtaSyntaxProfileMutation();
  const [updateProfile, { isLoading: isUpdating }] = useUpdateCtaSyntaxProfileMutation();
  const [deleteProfile, { isLoading: isDeleting }] = useDeleteCtaSyntaxProfileMutation();
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
  const [editingProfile, setEditingProfile] = useState<CtaSyntaxProfileDTO | null>(null);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const handleCreate = useCallback(async () => {
    if (isMutating || !newName.trim()) {
      return;
    }
    try {
      await createProfile({ name: newName.trim() }).unwrap();
      setNewName('');
      toast({ title: t('toast.profileCreated'), status: 'success' });
    } catch {
      toast({ title: t('toast.profileCreateFailed'), status: 'error' });
    }
  }, [createProfile, isMutating, newName, t, toast]);

  const handleUpdate = useCallback(
    async (profile: CtaSyntaxProfileDTO) => {
      if (isMutating) {
        return;
      }
      try {
        await updateProfile({
          id: profile.id,
          name: profile.name,
          spaces_to_underscores: profile.spaces_to_underscores,
          escape_parentheses: profile.escape_parentheses,
          escape_colons: profile.escape_colons,
          append_type_parentheses: profile.append_type_parentheses,
          prefix_artist_with_by: profile.prefix_artist_with_by,
        }).unwrap();
        setEditingProfile(null);
        toast({ title: t('toast.profileUpdated'), status: 'success' });
      } catch {
        toast({ title: t('toast.profileUpdateFailed'), status: 'error' });
      }
    },
    [isMutating, updateProfile, t, toast]
  );

  const handleDelete = useCallback(async () => {
    if (!pendingDeleteId || isMutating) {
      return;
    }
    const profileId = pendingDeleteId;
    setPendingDeleteId(null);
    try {
      await deleteProfile(profileId).unwrap();
      toast({ title: t('toast.profileDeleted'), status: 'success' });
    } catch {
      toast({ title: t('toast.profileDeleteFailed'), status: 'error' });
    }
  }, [deleteProfile, isMutating, pendingDeleteId, t, toast]);

  const overlayMessage = showRetry
    ? t('cta.ctaDataLoadFailed')
    : isMutating
      ? t('cta.savingCtaChanges')
      : profiles === undefined
        ? t('cta.loadingCtaData')
        : t('cta.refreshingCtaData');

  return (
    <Flex flexDir="column" gap={4} position="relative" minH="280px">
      <HStack>
        <FormControl isDisabled={isInteractionLocked}>
          <FormLabel>{t('cta.newProfileName')}</FormLabel>
          <Input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder={t('cta.profileNamePlaceholder')}
          />
        </FormControl>
        <Button onClick={handleCreate} isLoading={isCreating} isDisabled={isInteractionLocked || !newName.trim()}>
          {t('common.create')}
        </Button>
      </HStack>

      <VStack align="stretch" spacing={3}>
        {profiles?.map((profile) => (
          <Flex key={profile.id} p={3} bg="gray.700" borderRadius="md" flexDir="column" gap={2}>
            <HStack>
              <Text flex={1} fontWeight="bold">
                {profile.name}
              </Text>
              <IconButton
                aria-label={t('common.edit')}
                icon={<span>✏️</span>}
                size="sm"
                variant="ghost"
                isDisabled={isInteractionLocked}
                onClick={() => setEditingProfile(profile)}
              />
              <IconButton
                aria-label={t('common.delete')}
                icon={<span>🗑️</span>}
                size="sm"
                variant="ghost"
                colorScheme="red"
                isDisabled={isInteractionLocked}
                onClick={() => setPendingDeleteId(profile.id)}
              />
            </HStack>

            {editingProfile?.id === profile.id && (
              <SimpleGrid columns={2} spacing={2}>
                <FormControl isDisabled={isInteractionLocked}>
                  <FormLabel>{t('common.name')}</FormLabel>
                  <Input
                    value={editingProfile.name}
                    onChange={(e) => setEditingProfile({ ...editingProfile, name: e.target.value })}
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" isDisabled={isInteractionLocked}>
                  <FormLabel mb={0}>{t('cta.spacesToUnderscores')}</FormLabel>
                  <Switch
                    isChecked={editingProfile.spaces_to_underscores}
                    onChange={(e) =>
                      setEditingProfile({
                        ...editingProfile,
                        spaces_to_underscores: e.target.checked,
                      })
                    }
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" isDisabled={isInteractionLocked}>
                  <FormLabel mb={0}>{t('cta.escapeParentheses')}</FormLabel>
                  <Switch
                    isChecked={editingProfile.escape_parentheses}
                    onChange={(e) =>
                      setEditingProfile({
                        ...editingProfile,
                        escape_parentheses: e.target.checked,
                      })
                    }
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" isDisabled={isInteractionLocked}>
                  <FormLabel mb={0}>{t('cta.escapeColons')}</FormLabel>
                  <Switch
                    isChecked={editingProfile.escape_colons}
                    onChange={(e) =>
                      setEditingProfile({
                        ...editingProfile,
                        escape_colons: e.target.checked,
                      })
                    }
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" isDisabled={isInteractionLocked}>
                  <FormLabel mb={0}>{t('cta.appendTypeParentheses')}</FormLabel>
                  <Switch
                    isChecked={editingProfile.append_type_parentheses}
                    onChange={(e) =>
                      setEditingProfile({
                        ...editingProfile,
                        append_type_parentheses: e.target.checked,
                      })
                    }
                  />
                </FormControl>
                <FormControl display="flex" alignItems="center" isDisabled={isInteractionLocked}>
                  <FormLabel mb={0}>{t('cta.prefixArtistWithBy')}</FormLabel>
                  <Switch
                    isChecked={editingProfile.prefix_artist_with_by}
                    onChange={(e) =>
                      setEditingProfile({
                        ...editingProfile,
                        prefix_artist_with_by: e.target.checked,
                      })
                    }
                  />
                </FormControl>
                <HStack>
                  <Button
                    size="sm"
                    isLoading={isUpdating}
                    isDisabled={isInteractionLocked || !editingProfile.name.trim()}
                    onClick={() => handleUpdate(editingProfile)}
                  >
                    {t('common.save')}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    isDisabled={isInteractionLocked}
                    onClick={() => setEditingProfile(null)}
                  >
                    {t('common.cancel')}
                  </Button>
                </HStack>
              </SimpleGrid>
            )}
          </Flex>
        ))}
      </VStack>
      <CtaManagerOverlay
        isOpen={isLoading || isFetching || isMutating || isError}
        message={overlayMessage}
        onRetry={showRetry ? () => void refetch() : undefined}
      />
      <ConfirmationAlertDialog
        title={t('cta.deleteSyntaxProfile')}
        isOpen={pendingDeleteId !== null}
        onClose={() => setPendingDeleteId(null)}
        cancelButtonText={t('common.cancel')}
        acceptButtonText={t('common.delete')}
        acceptCallback={() => void handleDelete()}
        cancelCallback={() => setPendingDeleteId(null)}
        useInert={false}
      >
        <Text>{t('cta.deleteSyntaxProfileWarning')}</Text>
      </ConfirmationAlertDialog>
    </Flex>
  );
};

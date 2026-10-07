import type { ClipModelConfig, ClipTagSet } from '@features/cliptags/core/types';

import { Checkbox, createListCollection, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { setClipModelConfig } from '@features/cliptags/data/api';
import { clipTagKeys } from '@features/cliptags/data/keys';
import {
  clipModelConfigOptions,
  clipSyntaxProfilesOptions,
  clipTagSetsOptions,
  clipTagStatusOptions,
} from '@features/cliptags/data/queries';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { Select } from '@platform/ui/Select';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** The select has no empty value, so "no profile" is a value of its own. */
const NO_PROFILE = 'none';

/**
 * Which tag sets a model searches and how its tags are written into the prompt. Shown only where the tag database
 * works; every change is saved at once and the server's answer is what stays on screen.
 */
export const ClipTagModelSettings = ({ modelKey }: { modelKey: string }) => {
  const { t } = useTranslation();
  const status = useQuery(clipTagStatusOptions());
  const isAvailable = status.data?.available === true;
  const config = useQuery({ ...clipModelConfigOptions(modelKey), enabled: isAvailable });
  const profiles = useQuery({ ...clipSyntaxProfilesOptions(), enabled: isAvailable });
  const tagSets = useQuery({ ...clipTagSetsOptions(), enabled: isAvailable });
  const { refetch: refetchConfig } = config;
  const { refetch: refetchProfiles } = profiles;
  const { refetch: refetchTagSets } = tagSets;
  const retry = useCallback(
    () => void Promise.all([refetchConfig(), refetchProfiles(), refetchTagSets()]),
    [refetchConfig, refetchProfiles, refetchTagSets]
  );
  const profileOptions = useMemo(
    () => (profiles.data ?? []).map((profile) => ({ label: profile.name, value: profile.id })),
    [profiles.data]
  );

  if (!isAvailable) {
    return null;
  }

  const failed = config.isError || profiles.isError || tagSets.isError;

  if (failed) {
    return (
      <Stack align="start" gap="2">
        <Text color="fg.error" fontSize="md" role="alert">
          {t('cliptags.model.loadFailed')}
        </Text>
        <Button size="sm" variant="outline" onClick={retry}>
          {t('common.retry')}
        </Button>
      </Stack>
    );
  }

  if (!config.data || !profiles.data || !tagSets.data) {
    return (
      <HStack color="fg.muted" gap="2">
        <Spinner size="md" />
        <Text fontSize="md">{t('common.loading')}</Text>
      </HStack>
    );
  }

  return <ModelConfigForm config={config.data} modelKey={modelKey} profiles={profileOptions} tagSets={tagSets.data} />;
};

const ModelConfigForm = ({
  config,
  modelKey,
  profiles,
  tagSets,
}: {
  config: ClipModelConfig;
  modelKey: string;
  profiles: { label: string; value: string }[];
  tagSets: ClipTagSet[];
}) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const save = useMutation({
    mutationFn: (next: { syntaxProfileId: string | null; tagSetIds: string[] }) => setClipModelConfig(modelKey, next),
    // A refusal leaves the stored answer on screen; refetch in case something else changed it.
    onError: () => queryClient.invalidateQueries({ queryKey: clipTagKeys.modelConfig(modelKey) }),
    onSuccess: (saved) => {
      queryClient.setQueryData(clipTagKeys.modelConfig(modelKey), saved);
      // Prompt searches render and rank by this configuration, and the tag sets count the models using them.
      void queryClient.invalidateQueries({ queryKey: clipTagKeys.searches() });
      void queryClient.invalidateQueries({ queryKey: clipTagKeys.tagSets() });
    },
  });
  // While a save is in flight the user's choice shows, so the control does not jump back and forth.
  const shown = save.isPending ? save.variables : config;
  const profileItems = useMemo(
    () => [{ label: t('cliptags.model.noProfile'), value: NO_PROFILE }, ...profiles],
    [profiles, t]
  );
  const profileCollection = useMemo(() => createListCollection({ items: profileItems }), [profileItems]);
  const profileValue = useMemo(() => [shown.syntaxProfileId ?? NO_PROFILE], [shown.syntaxProfileId]);
  const selectedTagSetIds = new Set(shown.tagSetIds);

  const handleProfileChange = useCallback(
    ({ value }: { value: string[] }) => {
      const next = value[0];

      if (next !== undefined) {
        save.mutate({ syntaxProfileId: next === NO_PROFILE ? null : next, tagSetIds: shown.tagSetIds });
      }
    },
    [save, shown.tagSetIds]
  );
  const toggleTagSet = useCallback(
    (tagSetId: string, checked: boolean) => {
      const without = shown.tagSetIds.filter((id) => id !== tagSetId);

      save.mutate({ syntaxProfileId: shown.syntaxProfileId, tagSetIds: checked ? [...without, tagSetId] : without });
    },
    [save, shown.syntaxProfileId, shown.tagSetIds]
  );

  return (
    <Stack gap="3">
      <Text color="fg" fontSize="lg" fontWeight="500">
        {t('cliptags.model.title')}
      </Text>
      <Stack gap="1">
        <Text color="fg.muted" fontSize="md" id={`${modelKey}-cliptags-profile`}>
          {t('cliptags.model.syntaxProfile')}
        </Text>
        <Select
          aria-labelledby={`${modelKey}-cliptags-profile`}
          collection={profileCollection}
          size="lg"
          value={profileValue}
          onValueChange={handleProfileChange}
        />
      </Stack>
      <Stack gap="2" role="group" aria-label={t('cliptags.model.tagSets')}>
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.model.tagSets')}
        </Text>
        {tagSets.length === 0 ? (
          <Text color="fg.subtle" fontSize="md">
            {t('cliptags.model.noTagSets')}
          </Text>
        ) : (
          tagSets.map((tagSet) => (
            <TagSetCheckbox
              key={tagSet.id}
              checked={selectedTagSetIds.has(tagSet.id)}
              tagSet={tagSet}
              onToggle={toggleTagSet}
            />
          ))
        )}
      </Stack>
      {save.isError ? (
        <Text color="fg.error" fontSize="md" role="alert">
          {getApiErrorMessage(save.error, t('cliptags.model.saveFailed'))}
        </Text>
      ) : null}
    </Stack>
  );
};

const TagSetCheckbox = ({
  checked,
  onToggle,
  tagSet,
}: {
  checked: boolean;
  tagSet: ClipTagSet;
  onToggle: (tagSetId: string, checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => onToggle(tagSet.id, event.checked === true),
    [onToggle, tagSet.id]
  );

  return (
    <Checkbox.Root checked={checked} onCheckedChange={handleCheckedChange}>
      <Checkbox.HiddenInput />
      <Checkbox.Control />
      <Checkbox.Label color="fg" fontSize="md">
        {tagSet.name}
      </Checkbox.Label>
    </Checkbox.Root>
  );
};

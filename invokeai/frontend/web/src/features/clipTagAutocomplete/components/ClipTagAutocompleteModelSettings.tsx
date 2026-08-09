import { Checkbox, Flex, FormControl, FormLabel, Select, Spinner, Text, useToast } from '@invoke-ai/ui-library';
import { useIsAdmin } from 'features/auth/hooks/useIsAdmin';
import type { ChangeEvent } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetCtaModelConfigQuery,
  useListCtaSyntaxProfilesQuery,
  useListCtaTagSetsQuery,
  useSetCtaModelConfigMutation,
} from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';

type Props = {
  modelId: string;
};

export const ClipTagAutocompleteModelSettings = ({ modelId }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const isAdmin = useIsAdmin();
  const queryOptions = { skip: !isAdmin, refetchOnFocus: false, refetchOnReconnect: false };
  const { data: config, isLoading: configLoading, refetch } = useGetCtaModelConfigQuery(modelId, queryOptions);
  const { data: profiles } = useListCtaSyntaxProfilesQuery(undefined, queryOptions);
  const { data: tagSets } = useListCtaTagSetsQuery(undefined, queryOptions);
  const [setModelConfig, { isLoading: isSaving }] = useSetCtaModelConfigMutation();

  const save = useCallback(
    async (profileId: string, setIds: Set<string>): Promise<boolean> => {
      try {
        await setModelConfig({
          modelId,
          body: { syntax_profile_id: profileId || null, tag_set_ids: [...setIds] },
        }).unwrap();
        return true;
      } catch {
        toast({ title: t('cta.modelConfigSaveFailed'), status: 'error' });
        refetch();
        return false;
      }
    },
    [modelId, refetch, setModelConfig, t, toast]
  );

  if (!isAdmin) {
    return null;
  }
  if (configLoading || !config) {
    return <Spinner size="sm" />;
  }

  return (
    <ModelSettingsForm
      key={`${modelId}:${config.syntax_profile_id ?? ''}:${(config.tag_set_ids ?? []).join(',')}`}
      config={config}
      profiles={profiles ?? []}
      tagSets={tagSets ?? []}
      isSaving={isSaving}
      onSave={save}
    />
  );
};

type ModelSettingsFormProps = {
  config: S['CtaModelConfigDTO'];
  profiles: S['CtaSyntaxProfileDTO'][];
  tagSets: S['CtaTagSetDTO'][];
  isSaving: boolean;
  onSave: (profileId: string, setIds: Set<string>) => Promise<boolean>;
};

const ModelSettingsForm = ({ config, profiles, tagSets, isSaving, onSave }: ModelSettingsFormProps) => {
  const { t } = useTranslation();
  const [selectedProfile, setSelectedProfile] = useState(config.syntax_profile_id ?? '');
  const [selectedTagSets, setSelectedTagSets] = useState<Set<string>>(new Set(config.tag_set_ids ?? []));

  const handleProfileChange = useCallback(
    async (event: ChangeEvent<HTMLSelectElement>) => {
      const previous = selectedProfile;
      const profileId = event.target.value;
      setSelectedProfile(profileId);
      if (!(await onSave(profileId, selectedTagSets))) {
        setSelectedProfile(previous);
      }
    },
    [onSave, selectedProfile, selectedTagSets]
  );

  const handleTagSetToggle = useCallback(
    async (setId: string) => {
      const previous = selectedTagSets;
      const next = new Set(previous);
      if (next.has(setId)) {
        next.delete(setId);
      } else {
        next.add(setId);
      }
      setSelectedTagSets(next);
      if (!(await onSave(selectedProfile, next))) {
        setSelectedTagSets(previous);
      }
    },
    [onSave, selectedProfile, selectedTagSets]
  );

  return (
    <Flex flexDir="column" gap={4} opacity={isSaving ? 0.7 : 1}>
      <Text fontWeight="bold">{t('cta.clipTagAutocomplete')}</Text>
      <FormControl isDisabled={isSaving}>
        <FormLabel>{t('cta.syntaxProfile')}</FormLabel>
        <Select value={selectedProfile} onChange={(event) => void handleProfileChange(event)}>
          <option value="">{t('common.none')}</option>
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </Select>
      </FormControl>
      <FormControl isDisabled={isSaving}>
        <FormLabel>{t('cta.tagSets')}</FormLabel>
        <Flex flexDir="column" gap={2}>
          {tagSets.map((tagSet) => (
            <Checkbox
              key={tagSet.id}
              isChecked={selectedTagSets.has(tagSet.id)}
              onChange={() => void handleTagSetToggle(tagSet.id)}
            >
              <Text fontSize="sm">{tagSet.name}</Text>
            </Checkbox>
          ))}
        </Flex>
      </FormControl>
    </Flex>
  );
};

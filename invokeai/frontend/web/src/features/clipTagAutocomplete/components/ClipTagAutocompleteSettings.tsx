import { Button, Flex, FormControl, FormLabel, Select, Switch, Text, useToast } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getPromptPrefixOwner } from 'features/clipTagAutocomplete/util/promptPrefixRegistry';
import {
  clipTagAutocompleteEnabledChanged,
  clipTagAutocompleteHotPrefixChanged,
  selectClipTagAutocompleteEnabled,
  selectClipTagAutocompleteHotPrefix,
} from 'features/system/store/systemSlice';
import type { ChangeEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetCtaStatusQuery } from 'services/api/endpoints/clipTagAutocomplete';

type Props = {
  onOpenManager: () => void;
  isAdmin: boolean;
  isSettingsOpen: boolean;
};

export const ClipTagAutocompleteSettings = ({ onOpenManager, isAdmin, isSettingsOpen }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const dispatch = useAppDispatch();
  const enabled = useAppSelector(selectClipTagAutocompleteEnabled);
  const hotPrefix = useAppSelector(selectClipTagAutocompleteHotPrefix);
  const { data: status } = useGetCtaStatusQuery(undefined, {
    skip: !isSettingsOpen,
    refetchOnFocus: false,
    refetchOnReconnect: false,
  });
  const selectedPrefixIsOccupied = hotPrefix !== null && getPromptPrefixOwner(hotPrefix) !== null;

  const handleEnabledChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const newEnabled = e.target.checked;

      if (newEnabled) {
        // Determine which prefix to use
        const prefixToUse = hotPrefix ?? '~';

        // Check if prefix is available
        const owner = getPromptPrefixOwner(prefixToUse);
        if (owner !== null) {
          // Prefix is occupied — keep the selected prefix visible but disable CTA
          dispatch(clipTagAutocompleteEnabledChanged(false));
          toast({ title: t('cta.hotPrefixOccupied'), status: 'error' });
          return;
        }

        if (status?.available !== true) {
          dispatch(clipTagAutocompleteEnabledChanged(false));
          return;
        }

        dispatch(clipTagAutocompleteHotPrefixChanged(prefixToUse));
        dispatch(clipTagAutocompleteEnabledChanged(true));
      } else {
        dispatch(clipTagAutocompleteEnabledChanged(false));
      }
    },
    [dispatch, hotPrefix, status?.available, t, toast]
  );

  const handlePrefixChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const newPrefix = e.target.value as '~' | '*' | null;
      dispatch(clipTagAutocompleteHotPrefixChanged(newPrefix));
    },
    [dispatch]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl>
        <FormLabel>{t('settings.enableClipTagAutocomplete')}</FormLabel>
        <Switch
          isChecked={enabled}
          onChange={handleEnabledChange}
          isDisabled={status?.available !== true && !enabled}
        />
      </FormControl>

      <FormControl isInvalid={selectedPrefixIsOccupied}>
        <FormLabel>{t('settings.hotPrefix')}</FormLabel>
        <Select value={hotPrefix ?? ''} onChange={handlePrefixChange} isDisabled={enabled}>
          <option value="~">~ (tilde)</option>
          <option value="*">* (asterisk)</option>
        </Select>
      </FormControl>

      {status && !status.available && (
        <Text fontSize="sm" color="red.500">
          {status.reason === 'fts5_unavailable' && t('cta.fts5Unavailable')}
          {status.reason === 'database_incompatible' && t('cta.databaseIncompatible')}
          {status.reason === 'database_error' && t('cta.databaseError')}
        </Text>
      )}

      {isAdmin && (
        <Button onClick={onOpenManager} variant="outline" isDisabled={status?.available !== true}>
          {t('settings.manageCtaData')}
        </Button>
      )}
    </Flex>
  );
};

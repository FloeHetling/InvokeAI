import { HStack, Stack, Text } from '@chakra-ui/react';
import { useClipTagStatus } from '@features/cliptags/react';
import { useCapabilities } from '@features/identity';
import { Button } from '@platform/ui/Button';
import { useNavigate } from '@tanstack/react-router';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Availability of the tag database and the way to its manager. The switch and the hot prefix are ordinary
 * preference fields above this one.
 */
export const ClipTagSettings = () => {
  const { t } = useTranslation();
  const { canManageClipTags } = useCapabilities();
  const navigate = useNavigate();
  const { data: status, isError } = useClipTagStatus();
  const openManager = useCallback(() => void navigate({ to: '/cliptags' }), [navigate]);
  const isUnavailable = isError || (status !== undefined && !status.available);

  return (
    <Stack gap="3" w="full">
      {isUnavailable ? (
        <Text color="fg.error" fontSize="md" role="alert">
          {t(`cliptags.unavailable.${status?.reason ?? 'default'}`)}
        </Text>
      ) : null}
      {canManageClipTags ? (
        <HStack>
          <Button disabled={status === undefined || isUnavailable} size="sm" variant="outline" onClick={openManager}>
            {t('settings.catalog.clipTagManager.open')}
          </Button>
        </HStack>
      ) : null}
    </Stack>
  );
};

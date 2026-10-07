/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Box, Center, Icon, Spinner } from '@chakra-ui/react';
import { clipTagStatusOptions } from '@features/cliptags/data/queries';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { EmptyState } from '@platform/ui/EmptyState';
import { PageShell } from '@platform/ui/PageShell';
import { Tabs } from '@platform/ui/Tabs';
import { useIsMutating, useQuery } from '@tanstack/react-query';
import { DatabaseZapIcon, TriangleAlertIcon, UploadIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { BeforeUnloadGuard } from './BeforeUnloadGuard';
import { CLIP_TAG_WRITE_KEY } from './clipTagWrites';
import { ImportDialog } from './ImportDialog';
import { SyntaxProfilesTab } from './SyntaxProfilesTab';
import { TagSetsTab } from './TagSetsTab';
import { TagsTab } from './TagsTab';

type ManagerTab = 'syntaxProfiles' | 'tagSets' | 'tags';

const CONTENT_SX = {
  display: 'flex',
  flex: '1',
  flexDirection: 'column',
  maxW: '6xl',
  minH: '0',
  mx: 'auto',
  pb: '4',
  px: { base: 4, md: 8 },
  w: 'full',
} as const;
const TAB_CONTENT_SX = { display: 'flex', flex: '1', flexDirection: 'column', minH: '0', pb: '0', pt: '3' } as const;

/** The administrator's manager for CLIP tag autocomplete: tags, tag sets, syntax profiles and CSV import. */
export const ClipTagsPage = () => {
  const { t } = useTranslation();
  const status = useQuery(clipTagStatusOptions());
  const [isImportOpen, setIsImportOpen] = useState(false);
  const isAvailable = status.data?.available === true;
  const isWriting = useIsMutating({ mutationKey: CLIP_TAG_WRITE_KEY }) > 0;

  const openImport = useCallback(() => setIsImportOpen(true), []);
  const closeImport = useCallback(() => setIsImportOpen(false), []);

  return (
    <PageShell
      actions={
        isAvailable ? (
          <Button variant="solid" onClick={openImport}>
            <UploadIcon />
            {t('cliptags.manager.import.open')}
          </Button>
        ) : null
      }
      description={t('cliptags.manager.description')}
      scroll="content"
      title={t('cliptags.manager.title')}
    >
      {status.isPending ? (
        <Center aria-label={t('common.loading')} flex="1" role="status">
          <Spinner color="fg.subtle" size="lg" />
        </Center>
      ) : status.isError ? (
        <EmptyState
          danger
          description={getApiErrorMessage(status.error, '')}
          icon={<Icon as={TriangleAlertIcon} />}
          title={t('cliptags.manager.loadFailed')}
        >
          <Button variant="outline" onClick={() => void status.refetch()}>
            {t('common.retry')}
          </Button>
        </EmptyState>
      ) : !isAvailable ? (
        <EmptyState
          description={
            status.data.reason
              ? t(`cliptags.unavailable.${status.data.reason}`)
              : t('cliptags.manager.unavailableDescription')
          }
          icon={<Icon as={DatabaseZapIcon} />}
          title={t('cliptags.manager.unavailable')}
        />
      ) : (
        <Box css={CONTENT_SX}>
          <Manager onImport={openImport} />
        </Box>
      )}
      {isAvailable ? <ImportDialog isOpen={isImportOpen} onClose={closeImport} /> : null}
      {isWriting ? <BeforeUnloadGuard /> : null}
    </PageShell>
  );
};

const Manager = ({ onImport }: { onImport: () => void }) => {
  const { t } = useTranslation();
  const [tab, setTab] = useState<ManagerTab>('tags');
  // Following a tag set's "View tags" restarts the Tags tab on that set; the other filters are dropped with it.
  const [tagsSeed, setTagsSeed] = useState<{ key: number; tagSetId?: string }>({ key: 0 });

  const handleTabChange = useCallback((event: { value: string }) => {
    if (event.value === 'tags' || event.value === 'tagSets' || event.value === 'syntaxProfiles') {
      setTab(event.value);
    }
  }, []);
  const handleViewTags = useCallback((tagSetId: string) => {
    setTagsSeed((current) => ({ key: current.key + 1, tagSetId }));
    setTab('tags');
  }, []);

  return (
    <Tabs.Root
      colorPalette="accent"
      display="flex"
      flex="1"
      flexDirection="column"
      lazyMount
      minH="0"
      size="lg"
      value={tab}
      variant="line"
      onValueChange={handleTabChange}
    >
      <Tabs.List flexShrink={0}>
        <Tabs.Trigger value="tags">{t('cliptags.manager.tabs.tags')}</Tabs.Trigger>
        <Tabs.Trigger value="tagSets">{t('cliptags.manager.tabs.tagSets')}</Tabs.Trigger>
        <Tabs.Trigger value="syntaxProfiles">{t('cliptags.manager.tabs.syntaxProfiles')}</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content css={TAB_CONTENT_SX} value="tags">
        <TagsTab key={tagsSeed.key} initialTagSetId={tagsSeed.tagSetId} onImport={onImport} />
      </Tabs.Content>
      <Tabs.Content css={TAB_CONTENT_SX} value="tagSets">
        <TagSetsTab onViewTags={handleViewTags} />
      </Tabs.Content>
      <Tabs.Content css={TAB_CONTENT_SX} value="syntaxProfiles">
        <SyntaxProfilesTab />
      </Tabs.Content>
    </Tabs.Root>
  );
};

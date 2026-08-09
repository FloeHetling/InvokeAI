import {
  Button,
  Flex,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@invoke-ai/ui-library';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ImportTagsModal } from './manager/ImportTagsModal';
import { SyntaxProfilesTab } from './manager/SyntaxProfilesTab';
import { TagSetsTab } from './manager/TagSetsTab';
import { TagsTab } from './manager/TagsTab';

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

export const ClipTagAutocompleteManagerModal = ({ isOpen, onClose }: Props) => {
  const { t } = useTranslation();
  const [activeTab, setManagerTab] = useState(0);
  const [isImportOpen, setIsImportOpen] = useState(false);
  const [initialTagSetId, setInitialTagSetId] = useState<string>();
  const [isMutationBusy, setIsMutationBusy] = useState(false);

  const handleClose = () => {
    if (!isMutationBusy) {
      onClose();
    }
  };

  return (
    <>
      <Modal isOpen={isOpen && !isImportOpen} onClose={handleClose} size="6xl" isCentered>
        <ModalOverlay />
        <ModalContent maxH="80vh">
          <ModalHeader>{t('settings.manageCtaData')}</ModalHeader>
          <ModalCloseButton isDisabled={isMutationBusy} />
          <ModalBody>
            <Tabs index={activeTab} onChange={setManagerTab} isLazy lazyBehavior="unmount">
              <TabList>
                <Tab isDisabled={isMutationBusy && activeTab !== 0}>{t('common.tags')}</Tab>
                <Tab isDisabled={isMutationBusy && activeTab !== 1}>{t('common.tagSets')}</Tab>
                <Tab isDisabled={isMutationBusy && activeTab !== 2}>{t('common.syntaxProfiles')}</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <TagsTab
                    initialTagSetId={initialTagSetId}
                    onImport={() => setIsImportOpen(true)}
                    onMutationBusyChange={setIsMutationBusy}
                  />
                </TabPanel>
                <TabPanel>
                  <TagSetsTab
                    onMutationBusyChange={setIsMutationBusy}
                    onViewTags={(tagSetId) => {
                      setInitialTagSetId(tagSetId);
                      setManagerTab(0);
                    }}
                  />
                </TabPanel>
                <TabPanel>
                  <SyntaxProfilesTab onMutationBusyChange={setIsMutationBusy} />
                </TabPanel>
              </TabPanels>
            </Tabs>
          </ModalBody>
          <ModalFooter>
            <Flex justifyContent="flex-end">
              <Button onClick={handleClose} isDisabled={isMutationBusy}>
                {t('common.close')}
              </Button>
            </Flex>
          </ModalFooter>
        </ModalContent>
      </Modal>
      <ImportTagsModal isOpen={isImportOpen} onClose={() => setIsImportOpen(false)} />
    </>
  );
};

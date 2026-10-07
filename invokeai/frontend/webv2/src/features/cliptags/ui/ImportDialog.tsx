import type { ClipImportDestination, ClipImportMapping, ClipTagSet } from '@features/cliptags/core/types';

import { Dialog, Portal } from '@chakra-ui/react';
import {
  cancelClipImport,
  commitClipImport,
  downloadClipSampleCsv,
  prepareClipImport,
  stageClipImport,
} from '@features/cliptags/data/api';
import { clipTagSetsOptions } from '@features/cliptags/data/queries';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useCallback, useReducer, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { notify, useClipTagWrite } from './clipTagWrites';
import {
  hasDuplicateColumns,
  importReducer,
  INITIAL_IMPORT_STATE,
  isImportInFlight,
  toImportDestination,
  toImportMapping,
  type ImportDestinationDraft,
  type ImportMappingDraft,
} from './importFlow';
import { ImportBusy, MappingStep, PreviewStep, ResultStep, SelectFileStep } from './ImportSteps';

const SAMPLE_FILE_NAME = 'cta_sample_tags.csv';
const IMPORT_STEP_KEY = ['cliptags', 'import'] as const;
const IGNORE = () => undefined;
const EMPTY_TAG_SETS: readonly ClipTagSet[] = [];

export interface ImportDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Imports tags from a CSV file in steps: choose a file, map its columns, preview the result, choose where the tags
 * go, and read the summary. Each opening starts a fresh flow; the staged server session is discarded whenever the
 * flow is abandoned.
 */
export const ImportDialog = ({ isOpen, onClose }: ImportDialogProps) => {
  const presence = useExitPresence(isOpen);

  return presence.isMounted ? (
    <ImportFlow key={presence.generation} isOpen={isOpen} onClose={onClose} onExited={presence.release} />
  ) : null;
};

const ImportFlow = ({ isOpen, onClose, onExited }: { isOpen: boolean; onClose: () => void; onExited: () => void }) => {
  const { t } = useTranslation();
  const [state, dispatch] = useReducer(importReducer, INITIAL_IMPORT_STATE);
  const [isDownloading, setIsDownloading] = useState(false);
  const tagSetsQuery = useQuery({ ...clipTagSetsOptions(), enabled: state.step === 'preview' });
  const stage = useMutation({ mutationFn: (file: File) => stageClipImport(file), mutationKey: IMPORT_STEP_KEY });
  const prepare = useMutation({
    mutationFn: ({ mapping, sessionId }: { mapping: ClipImportMapping; sessionId: string }) =>
      prepareClipImport(sessionId, mapping),
    mutationKey: IMPORT_STEP_KEY,
  });
  const commit = useClipTagWrite(
    ({ destination, sessionId }: { destination: ClipImportDestination; sessionId: string }) =>
      commitClipImport(sessionId, destination)
  );
  // The staged session lives on the server until committed or cancelled; this is the one place that knows it.
  const sessionRef = useRef<string | null>(null);

  const discardSession = useCallback(() => {
    const sessionId = sessionRef.current;

    sessionRef.current = null;

    if (sessionId) {
      cancelClipImport(sessionId).catch(IGNORE);
    }
  }, []);
  const isLiveRef = useRef(true);

  // The page can unmount mid-flow (a sign-out, a navigation); the session must not outlive the flow.
  useMountEffect(() => {
    isLiveRef.current = true;

    return () => {
      isLiveRef.current = false;
      discardSession();
    };
  });

  const inFlight = isImportInFlight(state);
  const handleClose = useCallback(() => {
    if (inFlight) {
      return;
    }

    discardSession();
    onClose();
  }, [discardSession, inFlight, onClose]);
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        handleClose();
      }
    },
    [handleClose]
  );

  const handleChooseFile = useCallback(
    async (file: File) => {
      dispatch({ type: 'stageStarted' });

      try {
        const staged = await stage.mutateAsync(file);

        // The flow ended while the file was uploading, so no cleanup will ever see this session.
        if (!isLiveRef.current) {
          cancelClipImport(staged.sessionId).catch(IGNORE);

          return;
        }

        sessionRef.current = staged.sessionId;
        dispatch({ stage: staged, type: 'stageSucceeded' });
      } catch (error) {
        dispatch({
          message: getApiErrorMessage(error, t('cliptags.manager.import.uploadFailed')),
          type: 'stageFailed',
        });
      }
    },
    [stage, t]
  );
  const handlePrepare = useCallback(async () => {
    if (state.step !== 'mapping' || hasDuplicateColumns(state.mapping)) {
      return;
    }

    const { mapping, stage: staged } = state;

    dispatch({ type: 'prepareStarted' });

    try {
      const preview = await prepare.mutateAsync({ mapping: toImportMapping(mapping), sessionId: staged.sessionId });

      dispatch({ preview, type: 'prepareSucceeded' });
    } catch (error) {
      dispatch({
        message: getApiErrorMessage(error, t('cliptags.manager.import.prepareFailed')),
        type: 'prepareFailed',
      });
    }
  }, [prepare, state, t]);
  const handleCommit = useCallback(async () => {
    if (state.step !== 'preview') {
      return;
    }

    const destination = toImportDestination(state.destination);

    if (!destination) {
      return;
    }

    const { sessionId } = state.stage;

    dispatch({ type: 'commitStarted' });

    try {
      const result = await commit.mutateAsync({ destination, sessionId });

      // A committed session is gone from the server; there is nothing left to cancel.
      sessionRef.current = null;
      dispatch({ result, type: 'commitSucceeded' });
    } catch (error) {
      dispatch({ message: getApiErrorMessage(error, t('cliptags.manager.import.commitFailed')), type: 'commitFailed' });
    }
  }, [commit, state, t]);
  const handleChooseAnother = useCallback(() => {
    discardSession();
    dispatch({ type: 'reset' });
  }, [discardSession]);
  const handleDownloadSample = useCallback(async () => {
    setIsDownloading(true);

    try {
      downloadBlob(await downloadClipSampleCsv(), SAMPLE_FILE_NAME);
    } catch (error) {
      notify('error', t('cliptags.manager.import.sampleFailed'), getApiErrorMessage(error, ''));
    } finally {
      setIsDownloading(false);
    }
  }, [t]);
  const handleMappingChange = useCallback(
    (changes: Partial<ImportMappingDraft>) => dispatch({ changes, type: 'mappingChanged' }),
    []
  );
  const handleDestinationChange = useCallback(
    (changes: Partial<ImportDestinationDraft>) => dispatch({ changes, type: 'destinationChanged' }),
    []
  );
  const reopenMapping = useCallback(() => dispatch({ type: 'mappingReopened' }), []);

  const destination = state.step === 'preview' ? toImportDestination(state.destination) : null;
  const canCommit = state.step === 'preview' && destination !== null && state.preview.summary.validRows > 0;

  return (
    <Dialog.Root
      closeOnEscape={!inFlight}
      closeOnInteractOutside={!inFlight}
      lazyMount
      open={isOpen}
      scrollBehavior="inside"
      size="md"
      unmountOnExit
      onExitComplete={onExited}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Dialog.Title>{t('cliptags.manager.import.title')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              {state.step === 'select' ? (
                <SelectFileStep
                  error={state.error}
                  isDownloading={isDownloading}
                  onChooseFile={handleChooseFile}
                  onDownloadSample={handleDownloadSample}
                />
              ) : null}
              {state.step === 'staging' ? <ImportBusy label={t('cliptags.manager.import.uploading')} /> : null}
              {state.step === 'mapping' ? (
                <MappingStep
                  error={state.error}
                  mapping={state.mapping}
                  stage={state.stage}
                  onChange={handleMappingChange}
                />
              ) : null}
              {state.step === 'preparing' ? <ImportBusy label={t('cliptags.manager.import.validating')} /> : null}
              {state.step === 'preview' ? (
                <PreviewStep
                  destination={state.destination}
                  error={state.error}
                  preview={state.preview}
                  tagSets={tagSetsQuery.data ?? EMPTY_TAG_SETS}
                  onChange={handleDestinationChange}
                />
              ) : null}
              {state.step === 'committing' ? <ImportBusy label={t('cliptags.manager.import.importing')} /> : null}
              {state.step === 'result' ? <ResultStep result={state.result} /> : null}
            </Dialog.Body>
            <Dialog.Footer>
              {state.step === 'mapping' || state.step === 'preview' ? (
                <Button
                  me="auto"
                  variant="ghost"
                  onClick={state.step === 'mapping' ? handleChooseAnother : reopenMapping}
                >
                  {state.step === 'mapping'
                    ? t('cliptags.manager.import.chooseAnother')
                    : t('cliptags.manager.import.back')}
                </Button>
              ) : null}
              {state.step === 'result' ? (
                <Button variant="solid" onClick={handleClose}>
                  {t('common.done')}
                </Button>
              ) : (
                <Button disabled={inFlight} variant="ghost" onClick={handleClose}>
                  {t('common.cancel')}
                </Button>
              )}
              {state.step === 'mapping' ? (
                <Button disabled={hasDuplicateColumns(state.mapping)} variant="solid" onClick={handlePrepare}>
                  {t('cliptags.manager.import.previewImport')}
                </Button>
              ) : null}
              {state.step === 'preview' ? (
                <Button disabled={!canCommit} variant="solid" onClick={handleCommit}>
                  {t('common.import')}
                </Button>
              ) : null}
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={inFlight} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

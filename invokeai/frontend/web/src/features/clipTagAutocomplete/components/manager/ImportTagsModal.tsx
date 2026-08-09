import {
  Button,
  FormControl,
  FormLabel,
  HStack,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Progress,
  Select,
  Switch,
  Text,
  useToast,
  VStack,
} from '@invoke-ai/ui-library';
import { useCtaBeforeUnload } from 'features/clipTagAutocomplete/hooks/useCtaBeforeUnload';
import { type ChangeEvent, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useCancelCtaImportMutation,
  useCommitCtaImportMutation,
  useDownloadCtaSampleMutation,
  useListCtaTagSetsQuery,
  usePrepareCtaImportMutation,
  useStageCtaImportMutation,
} from 'services/api/endpoints/clipTagAutocomplete';
import type { S } from 'services/api/types';

type Props = {
  isOpen: boolean;
  onClose: () => void;
};

type ImportStep = 'select_file' | 'uploading' | 'mapping' | 'preparing' | 'preview' | 'committing' | 'result' | 'error';
type Destination = 'uncategorized' | 'new_set' | 'existing_set';
type ImportSummary = S['CtaImportSummaryDTO'] | S['CtaImportResultDTO'];

const NO_COLUMN = -1;

export const ImportTagsModal = ({ isOpen, onClose }: Props) => {
  const { t } = useTranslation();
  const toast = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [stageImport] = useStageCtaImportMutation();
  const [prepareImport] = usePrepareCtaImportMutation();
  const [commitImport] = useCommitCtaImportMutation();
  const [cancelImport] = useCancelCtaImportMutation();
  const [downloadSample, { isLoading: isDownloadingSample }] = useDownloadCtaSampleMutation();
  const { data: tagSets } = useListCtaTagSetsQuery(undefined, { skip: !isOpen });

  const [step, setStep] = useState<ImportStep>('select_file');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [stage, setStage] = useState<S['CtaImportStageDTO'] | null>(null);
  const [preview, setPreview] = useState<S['CtaImportPreviewRow'][]>([]);
  const [summary, setSummary] = useState<ImportSummary | null>(null);
  const [delimiter, setDelimiter] = useState<S['CtaImportMapping']['delimiter']>(',');
  const [hasHeader, setHasHeader] = useState(true);
  const [tagColumn, setTagColumn] = useState(0);
  const [popularityColumn, setPopularityColumn] = useState(NO_COLUMN);
  const [typeColumn, setTypeColumn] = useState(NO_COLUMN);
  const [destination, setDestination] = useState<Destination>('uncategorized');
  const [newSetName, setNewSetName] = useState('');
  const [existingSetId, setExistingSetId] = useState('');
  const [importMode, setImportMode] = useState<'merge' | 'replace'>('merge');

  useCtaBeforeUnload(step === 'committing');

  const reset = useCallback(() => {
    setStep('select_file');
    setSessionId(null);
    setStage(null);
    setPreview([]);
    setSummary(null);
    setDestination('uncategorized');
    setNewSetName('');
    setExistingSetId('');
    setImportMode('merge');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const cancelStagedSession = useCallback(async () => {
    if (!sessionId || step === 'committing' || step === 'result') {
      return;
    }
    try {
      await cancelImport(sessionId).unwrap();
    } catch {
      // The server may already have expired or discarded the staged session.
    }
  }, [cancelImport, sessionId, step]);

  const handleClose = useCallback(() => {
    if (step === 'committing') {
      return;
    }
    void cancelStagedSession();
    reset();
    onClose();
  }, [cancelStagedSession, onClose, reset, step]);

  const handleFileSelect = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      setStep('uploading');
      const formData = new FormData();
      formData.append('file', file);
      try {
        const result = await stageImport(formData).unwrap();
        setSessionId(result.session_id);
        setStage(result);
        setDelimiter(result.detected_delimiter);
        setHasHeader(result.detected_header);
        const normalizedColumns = result.columns.map((column) => column.trim().toLowerCase());
        setTagColumn(
          Math.max(
            0,
            normalizedColumns.findIndex((column) => column === 'tag')
          )
        );
        setPopularityColumn(normalizedColumns.findIndex((column) => column === 'popularity'));
        setTypeColumn(normalizedColumns.findIndex((column) => column === 'type'));
        setStep('mapping');
      } catch {
        setStep('error');
        toast({ title: t('toast.importUploadFailed'), status: 'error' });
      }
    },
    [stageImport, t, toast]
  );

  const mappedColumns = useMemo(
    () => [tagColumn, popularityColumn, typeColumn].filter((column) => column !== NO_COLUMN),
    [popularityColumn, tagColumn, typeColumn]
  );
  const hasDuplicateMapping = new Set(mappedColumns).size !== mappedColumns.length;

  const handlePrepare = useCallback(async () => {
    if (!sessionId || !stage || hasDuplicateMapping) {
      return;
    }
    setStep('preparing');
    try {
      const result = await prepareImport({
        sessionId,
        body: {
          tag_column: tagColumn,
          popularity_column: stage.is_single_column || popularityColumn === NO_COLUMN ? null : popularityColumn,
          type_column: stage.is_single_column || typeColumn === NO_COLUMN ? null : typeColumn,
          delimiter,
          first_row_contains_column_names: hasHeader,
        },
      }).unwrap();
      setPreview(result.preview);
      setSummary(result.summary);
      setStep('preview');
    } catch {
      await cancelStagedSession();
      setSessionId(null);
      setStep('error');
      toast({ title: t('toast.importPrepareFailed'), status: 'error' });
    }
  }, [
    cancelStagedSession,
    delimiter,
    hasDuplicateMapping,
    hasHeader,
    popularityColumn,
    prepareImport,
    sessionId,
    stage,
    tagColumn,
    t,
    toast,
    typeColumn,
  ]);

  const canCommit =
    destination === 'uncategorized' ||
    (destination === 'new_set' && Boolean(newSetName.trim())) ||
    (destination === 'existing_set' && Boolean(existingSetId));

  const handleCommit = useCallback(async () => {
    if (!sessionId || !canCommit) {
      return;
    }
    setStep('committing');
    const importDestination =
      destination === 'new_set'
        ? ({ type: 'new_set', name: newSetName.trim() } as const)
        : destination === 'existing_set'
          ? ({ type: 'existing_set', tag_set_id: existingSetId, mode: importMode } as const)
          : ({ type: 'uncategorized' } as const);
    try {
      const result = await commitImport({ sessionId, body: { destination: importDestination } }).unwrap();
      setSummary(result);
      setSessionId(null);
      setStep('result');
    } catch {
      setStep('preview');
      toast({ title: t('toast.importCommitFailed'), status: 'error' });
    }
  }, [canCommit, commitImport, destination, existingSetId, importMode, newSetName, sessionId, t, toast]);

  const handleDownloadSample = useCallback(async () => {
    try {
      const blob = await downloadSample().unwrap();
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = href;
      anchor.download = 'clip_tag_autocomplete_sample.csv';
      anchor.click();
      URL.revokeObjectURL(href);
    } catch {
      toast({ title: t('cta.sampleDownloadFailed'), status: 'error' });
    }
  }, [downloadSample, t, toast]);

  const getColumnLabel = useCallback(
    (columnIndex: number) =>
      hasHeader
        ? (stage?.columns[columnIndex] ?? `${columnIndex + 1}`)
        : t('cta.columnNumber', { count: columnIndex + 1 }),
    [hasHeader, stage?.columns, t]
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      size="xl"
      isCentered
      closeOnEsc={step !== 'committing'}
      closeOnOverlayClick={step !== 'committing'}
    >
      <ModalOverlay />
      <ModalContent maxH="80vh">
        <ModalHeader>{t('cta.importTags')}</ModalHeader>
        {step !== 'committing' && <ModalCloseButton />}
        <ModalBody overflowY="auto">
          {step === 'select_file' && (
            <VStack spacing={4}>
              <Text>{t('cta.selectCsvFile')}</Text>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <HStack>
                <Button onClick={() => fileInputRef.current?.click()}>{t('cta.chooseFile')}</Button>
                <Button variant="ghost" isLoading={isDownloadingSample} onClick={() => void handleDownloadSample()}>
                  {t('cta.downloadSampleCsv')}
                </Button>
              </HStack>
            </VStack>
          )}

          {step === 'uploading' && <BusyState label={t('cta.uploading')} />}

          {step === 'mapping' && stage && (
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel>{t('cta.delimiter')}</FormLabel>
                <Select value={delimiter} onChange={(event) => setDelimiter(event.target.value as typeof delimiter)}>
                  <option value=",">{t('cta.comma')}</option>
                  <option value=";">{t('cta.semicolon')}</option>
                  <option value={'\t'}>{t('cta.tab')}</option>
                  <option value="|">{t('cta.pipe')}</option>
                </Select>
              </FormControl>
              <FormControl display="flex" alignItems="center">
                <FormLabel mb={0}>{t('cta.firstRowHeader')}</FormLabel>
                <Switch isChecked={hasHeader} onChange={(event) => setHasHeader(event.target.checked)} />
              </FormControl>
              {!stage.is_single_column && (
                <>
                  <ColumnSelect
                    label={t('cta.tagColumn')}
                    value={tagColumn}
                    columns={stage.columns}
                    getColumnLabel={getColumnLabel}
                    onChange={setTagColumn}
                    isRequired
                  />
                  <ColumnSelect
                    label={t('cta.popularityColumn')}
                    value={popularityColumn}
                    columns={stage.columns}
                    getColumnLabel={getColumnLabel}
                    onChange={setPopularityColumn}
                  />
                  <ColumnSelect
                    label={t('cta.typeColumn')}
                    value={typeColumn}
                    columns={stage.columns}
                    getColumnLabel={getColumnLabel}
                    onChange={setTypeColumn}
                  />
                </>
              )}
              <VStack align="stretch" maxH="160px" overflowY="auto" spacing={1}>
                {stage.sample_rows.map((row, rowIndex) => (
                  <Text key={rowIndex} fontSize="xs" color="gray.400">
                    {row.join(' | ')}
                  </Text>
                ))}
              </VStack>
              {hasDuplicateMapping && <Text color="error.500">{t('cta.columnsMustBeUnique')}</Text>}
              <Button isDisabled={hasDuplicateMapping} onClick={() => void handlePrepare()}>
                {t('cta.previewImport')}
              </Button>
            </VStack>
          )}

          {step === 'preparing' && <BusyState label={t('cta.validatingFile')} />}

          {step === 'preview' && summary && (
            <VStack spacing={4} align="stretch">
              <Text fontWeight="bold">{t('cta.preview')}</Text>
              <VStack align="stretch" maxH="200px" overflowY="auto" spacing={1}>
                {preview.map((row, index) => (
                  <HStack key={`${row.canonical_content}-${row.tag_type}-${index}`} fontSize="sm">
                    <Text flex={1}>{row.canonical_content}</Text>
                    <Text color="gray.400">{row.tag_type}</Text>
                    <Text color="gray.400">{row.popularity ?? '—'}</Text>
                  </HStack>
                ))}
              </VStack>
              {'rows_read' in summary && (
                <Text fontSize="sm" color="gray.400">
                  {t('cta.rowsRead')}: {summary.rows_read} | {t('cta.validTags')}: {summary.valid_rows} |{' '}
                  {t('cta.skippedRows')}: {summary.skipped_rows}
                </Text>
              )}
              <FormControl>
                <FormLabel>{t('cta.destination')}</FormLabel>
                <Select value={destination} onChange={(event) => setDestination(event.target.value as Destination)}>
                  <option value="uncategorized">{t('cta.uncategorized')}</option>
                  <option value="new_set">{t('cta.newTagSet')}</option>
                  <option value="existing_set">{t('cta.existingTagSet')}</option>
                </Select>
              </FormControl>
              {destination === 'new_set' && (
                <FormControl>
                  <FormLabel>{t('cta.setName')}</FormLabel>
                  <Input value={newSetName} onChange={(event) => setNewSetName(event.target.value)} />
                </FormControl>
              )}
              {destination === 'existing_set' && (
                <>
                  <FormControl>
                    <FormLabel>{t('cta.selectTagSet')}</FormLabel>
                    <Select value={existingSetId} onChange={(event) => setExistingSetId(event.target.value)}>
                      <option value="">{t('cta.selectTagSet')}</option>
                      {tagSets?.map((tagSet) => (
                        <option key={tagSet.id} value={tagSet.id}>
                          {tagSet.name}
                        </option>
                      ))}
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel>{t('cta.importMode')}</FormLabel>
                    <Select
                      value={importMode}
                      onChange={(event) => setImportMode(event.target.value as typeof importMode)}
                    >
                      <option value="merge">{t('common.merge')}</option>
                      <option value="replace">{t('common.replace')}</option>
                    </Select>
                  </FormControl>
                </>
              )}
              <Button colorScheme="green" isDisabled={!canCommit} onClick={() => void handleCommit()}>
                {t('common.import')}
              </Button>
            </VStack>
          )}

          {step === 'committing' && <BusyState label={t('cta.importingTags')} />}

          {step === 'result' && summary && 'new_tags' in summary && (
            <VStack spacing={3} align="stretch">
              <Text fontWeight="bold">{t('cta.importComplete')}</Text>
              <Text>
                {t('cta.newTags')}: {summary.new_tags}
              </Text>
              <Text>
                {t('cta.existingMerged')}: {summary.existing_tags_merged}
              </Text>
              <Text>
                {t('cta.popularityUpdated')}: {summary.popularity_updated}
              </Text>
              <Text>
                {t('cta.conflictsIgnored')}: {summary.type_conflicts_ignored}
              </Text>
              <Text>
                {t('cta.skippedRows')}: {summary.skipped_rows}
              </Text>
            </VStack>
          )}

          {step === 'error' && (
            <VStack spacing={4}>
              <Text>{t('cta.importFailedDescription')}</Text>
              <Button
                onClick={() => {
                  reset();
                }}
              >
                {t('cta.chooseAnotherFile')}
              </Button>
            </VStack>
          )}
        </ModalBody>
        {step !== 'committing' && (
          <ModalFooter>
            <Button onClick={handleClose}>{t('common.close')}</Button>
          </ModalFooter>
        )}
      </ModalContent>
    </Modal>
  );
};

const BusyState = ({ label }: { label: string }) => (
  <VStack spacing={4}>
    <Text>{label}</Text>
    <Progress size="xs" isIndeterminate w="100%" />
  </VStack>
);

type ColumnSelectProps = {
  label: string;
  value: number;
  columns: string[];
  getColumnLabel: (columnIndex: number) => string;
  onChange: (columnIndex: number) => void;
  isRequired?: boolean;
};

const ColumnSelect = ({ label, value, columns, getColumnLabel, onChange, isRequired }: ColumnSelectProps) => {
  const { t } = useTranslation();
  return (
    <FormControl isRequired={isRequired}>
      <FormLabel>{label}</FormLabel>
      <Select value={value} onChange={(event) => onChange(Number(event.target.value))}>
        {!isRequired && <option value={NO_COLUMN}>{t('cta.ignoreColumn')}</option>}
        {columns.map((_column, columnIndex) => (
          <option key={columnIndex} value={columnIndex}>
            {getColumnLabel(columnIndex)}
          </option>
        ))}
      </Select>
    </FormControl>
  );
};

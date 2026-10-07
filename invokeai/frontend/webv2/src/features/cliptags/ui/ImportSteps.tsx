import type {
  ClipImportDelimiter,
  ClipImportPreview,
  ClipImportResult,
  ClipImportStage,
  ClipTagSet,
} from '@features/cliptags/core/types';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Box, HStack, Input, Progress, Stack, Switch, Table, Text } from '@chakra-ui/react';
import { formatCount } from '@platform/i18n/languages';
import { Button } from '@platform/ui/Button';
import { Field } from '@platform/ui/Field';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { ErrorAlert } from './ErrorAlert';
import {
  hasDuplicateColumns,
  IMPORT_DELIMITERS,
  NO_COLUMN,
  type ImportDestinationDraft,
  type ImportMappingDraft,
} from './importFlow';
import { OptionSelect, type Option } from './OptionSelect';
import { TagTypeBadge } from './tagTypes';

const IMPORT_FILE_ACCEPT = '.csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain';
const SWITCH_CHECKED_STYLES = { bg: 'accent.solid' };
const HEADER_SWITCH_PADDING = { base: '0', sm: '6' } as const;
const SAMPLE_CONTENT_PROPS = { minW: 'full' } as const;
const TABLE_CELL_PROPS = { fontSize: 'xs', maxW: '14rem', overflowWrap: 'anywhere', py: '1' } as const;

const DELIMITER_LABEL_KEYS: Record<ClipImportDelimiter, string> = {
  '\t': 'cliptags.manager.import.delimiters.tab',
  ',': 'cliptags.manager.import.delimiters.comma',
  ';': 'cliptags.manager.import.delimiters.semicolon',
  '|': 'cliptags.manager.import.delimiters.pipe',
};

/** A running request: a label over an indeterminate bar, announced politely. */
export const ImportBusy = ({ label }: { label: string }) => (
  <Stack gap="2" minH="24" justify="center" role="status">
    <Text fontSize="md">{label}</Text>
    <Progress.Root aria-label={label} size="md" value={null}>
      <Progress.Track>
        <Progress.Range />
      </Progress.Track>
    </Progress.Root>
  </Stack>
);

export const SelectFileStep = ({
  error,
  isDownloading,
  onChooseFile,
  onDownloadSample,
}: {
  error: string | null;
  isDownloading: boolean;
  onChooseFile: (file: File) => void;
  onDownloadSample: () => void;
}) => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const openPicker = useCallback(() => inputRef.current?.click(), []);
  const handleChange = useCallback(
    (event: { currentTarget: HTMLInputElement }) => {
      const file = event.currentTarget.files?.[0];

      // Clearing the value lets the same file be chosen again after a failure.
      event.currentTarget.value = '';

      if (file) {
        onChooseFile(file);
      }
    },
    [onChooseFile]
  );

  return (
    <Stack gap="4">
      <Stack gap="1">
        <Text fontSize="md">{t('cliptags.manager.import.selectFile')}</Text>
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.manager.import.selectFileHint')}
        </Text>
      </Stack>
      {error ? <ErrorAlert message={error} /> : null}
      <input ref={inputRef} accept={IMPORT_FILE_ACCEPT} hidden type="file" onChange={handleChange} />
      <HStack flexWrap="wrap" gap="2">
        <Button variant="solid" onClick={openPicker}>
          {t('cliptags.manager.import.chooseFile')}
        </Button>
        <Button loading={isDownloading} variant="ghost" onClick={onDownloadSample}>
          {t('cliptags.manager.import.downloadSample')}
        </Button>
      </HStack>
    </Stack>
  );
};

const getColumnLabel = (stage: ClipImportStage, mapping: ImportMappingDraft, index: number, fallback: string): string =>
  (mapping.firstRowContainsColumnNames ? stage.columns[index]?.trim() : '') || fallback;

export const MappingStep = ({
  error,
  mapping,
  onChange,
  stage,
}: {
  error: string | null;
  mapping: ImportMappingDraft;
  stage: ClipImportStage;
  onChange: (changes: Partial<ImportMappingDraft>) => void;
}) => {
  const { t } = useTranslation();
  const hasDuplicates = hasDuplicateColumns(mapping);
  const delimiterOptions = useMemo<Option[]>(
    () => IMPORT_DELIMITERS.map((delimiter) => ({ label: t(DELIMITER_LABEL_KEYS[delimiter]), value: delimiter })),
    [t]
  );
  const columnLabels = useMemo(
    () =>
      stage.columns.map((_name, index) =>
        getColumnLabel(stage, mapping, index, t('cliptags.manager.import.columnNumber', { count: index + 1 }))
      ),
    [mapping, stage, t]
  );
  const requiredOptions = useMemo<Option[]>(
    () => columnLabels.map((label, index) => ({ label, value: String(index) })),
    [columnLabels]
  );
  const optionalOptions = useMemo<Option[]>(
    () => [{ label: t('cliptags.manager.import.ignoreColumn'), value: String(NO_COLUMN) }, ...requiredOptions],
    [requiredOptions, t]
  );

  const handleDelimiterChange = useCallback(
    (value: string) => {
      const delimiter = IMPORT_DELIMITERS.find((candidate) => candidate === value);

      if (delimiter) {
        onChange({ delimiter });
      }
    },
    [onChange]
  );
  const handleHeaderChange = useCallback(
    (event: { checked: boolean }) => onChange({ firstRowContainsColumnNames: event.checked }),
    [onChange]
  );
  const handleTagColumnChange = useCallback((value: string) => onChange({ tagColumn: Number(value) }), [onChange]);
  const handlePopularityColumnChange = useCallback(
    (value: string) => onChange({ popularityColumn: Number(value) }),
    [onChange]
  );
  const handleTypeColumnChange = useCallback((value: string) => onChange({ typeColumn: Number(value) }), [onChange]);

  return (
    <Stack gap="4">
      {error ? <ErrorAlert message={error} /> : null}
      <HStack align="flex-start" flexWrap="wrap" gap="3">
        <Field flex="1 1 10rem" label={t('cliptags.manager.import.delimiter')}>
          <OptionSelect
            inDialog
            label={t('cliptags.manager.import.delimiter')}
            options={delimiterOptions}
            value={mapping.delimiter}
            onChange={handleDelimiterChange}
          />
        </Field>
        <Switch.Root
          alignItems="center"
          checked={mapping.firstRowContainsColumnNames}
          display="flex"
          flex="1 1 12rem"
          gap="3"
          justifyContent="space-between"
          pt={HEADER_SWITCH_PADDING}
          onCheckedChange={handleHeaderChange}
        >
          <Switch.Label color="fg" fontSize="md" m="0">
            {t('cliptags.manager.import.firstRowHeader')}
          </Switch.Label>
          <Switch.HiddenInput />
          <Switch.Control _checked={SWITCH_CHECKED_STYLES}>
            <Switch.Thumb />
          </Switch.Control>
        </Switch.Root>
      </HStack>
      {stage.isSingleColumn ? (
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.manager.import.singleColumn')}
        </Text>
      ) : (
        <Stack gap="3">
          <Field required label={t('cliptags.manager.import.tagColumn')}>
            <OptionSelect
              inDialog
              label={t('cliptags.manager.import.tagColumn')}
              options={requiredOptions}
              value={String(mapping.tagColumn)}
              onChange={handleTagColumnChange}
            />
          </Field>
          <HStack align="flex-start" flexWrap="wrap" gap="3">
            <Field flex="1 1 10rem" label={t('cliptags.manager.import.popularityColumn')}>
              <OptionSelect
                inDialog
                label={t('cliptags.manager.import.popularityColumn')}
                options={optionalOptions}
                value={String(mapping.popularityColumn)}
                onChange={handlePopularityColumnChange}
              />
            </Field>
            <Field flex="1 1 10rem" label={t('cliptags.manager.import.typeColumn')}>
              <OptionSelect
                inDialog
                label={t('cliptags.manager.import.typeColumn')}
                options={optionalOptions}
                value={String(mapping.typeColumn)}
                onChange={handleTypeColumnChange}
              />
            </Field>
          </HStack>
          {hasDuplicates ? <ErrorAlert message={t('cliptags.manager.import.columnsMustBeUnique')} /> : null}
        </Stack>
      )}
      <Stack gap="1.5">
        <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
          {t('cliptags.manager.import.sampleRows')}
        </Text>
        <Scrollable
          borderColor="border.subtle"
          borderWidth="1px"
          contentProps={SAMPLE_CONTENT_PROPS}
          label={t('cliptags.manager.import.sampleRows')}
          maxH="48"
          orientation="horizontal"
          rounded="md"
        >
          <Table.Root size="md">
            <Table.Header>
              <Table.Row bg="bg.muted">
                {columnLabels.map((label, index) => (
                  <Table.ColumnHeader key={index} {...TABLE_CELL_PROPS} color="fg.muted">
                    {label}
                  </Table.ColumnHeader>
                ))}
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {stage.sampleRows.map((row, rowIndex) => (
                <Table.Row key={rowIndex}>
                  {columnLabels.map((_label, index) => (
                    <Table.Cell key={index} {...TABLE_CELL_PROPS}>
                      {row[index] ?? ''}
                    </Table.Cell>
                  ))}
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Root>
        </Scrollable>
      </Stack>
    </Stack>
  );
};

const Stat = ({ label, value }: { label: string; value: number }) => (
  <Stack flex="1 1 6rem" gap="0">
    <Text fontSize="lg" fontVariantNumeric="tabular-nums" fontWeight="600">
      {formatCount(value)}
    </Text>
    <Text color="fg.muted" fontSize="xs">
      {label}
    </Text>
  </Stack>
);

export const PreviewStep = ({
  destination,
  error,
  onChange,
  preview,
  tagSets,
}: {
  destination: ImportDestinationDraft;
  error: string | null;
  preview: ClipImportPreview;
  tagSets: readonly ClipTagSet[];
  onChange: (changes: Partial<ImportDestinationDraft>) => void;
}) => {
  const { t } = useTranslation();
  const { summary } = preview;
  const destinationOptions = useMemo<Option[]>(
    () => [
      { label: t('cliptags.manager.import.destinations.uncategorized'), value: 'uncategorized' },
      { label: t('cliptags.manager.import.destinations.new_set'), value: 'new_set' },
      { label: t('cliptags.manager.import.destinations.existing_set'), value: 'existing_set' },
    ],
    [t]
  );
  const tagSetOptions = useMemo<Option[]>(
    () => tagSets.map((tagSet) => ({ label: tagSet.name, value: tagSet.id })),
    [tagSets]
  );
  const modeOptions = useMemo<Option[]>(
    () => [
      { label: t('cliptags.manager.import.modes.merge'), value: 'merge' },
      { label: t('cliptags.manager.import.modes.replace'), value: 'replace' },
    ],
    [t]
  );

  const handleKindChange = useCallback(
    (value: string) => {
      if (value === 'uncategorized' || value === 'new_set' || value === 'existing_set') {
        onChange({ kind: value });
      }
    },
    [onChange]
  );
  const handleNewSetNameChange = useCallback(
    (event: { currentTarget: { value: string } }) => onChange({ newSetName: event.currentTarget.value }),
    [onChange]
  );
  const handleModeChange = useCallback(
    (value: string) => onChange({ mode: value === 'replace' ? 'replace' : 'merge' }),
    [onChange]
  );

  return (
    <Stack gap="4">
      {error ? <ErrorAlert message={error} /> : null}
      <HStack
        bg="bg.subtle"
        borderColor="border.subtle"
        borderWidth="1px"
        flexWrap="wrap"
        gap="3"
        px="3"
        py="2"
        rounded="md"
      >
        <Stat label={t('cliptags.manager.import.rowsRead')} value={summary.rowsRead} />
        <Stat label={t('cliptags.manager.import.validTags')} value={summary.validRows} />
        <Stat label={t('cliptags.manager.import.skippedRows')} value={summary.skippedRows} />
      </HStack>
      {summary.validRows === 0 ? <ErrorAlert message={t('cliptags.manager.import.noValidRows')} /> : null}
      {summary.invalidPopularityToUnknown > 0 ? (
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.manager.import.invalidPopularity', {
            count: summary.invalidPopularityToUnknown,
            total: formatCount(summary.invalidPopularityToUnknown),
          })}
        </Text>
      ) : null}
      {summary.unknownTypesToOther > 0 ? (
        <Text color="fg.muted" fontSize="md">
          {t('cliptags.manager.import.unknownTypes', {
            count: summary.unknownTypesToOther,
            total: formatCount(summary.unknownTypesToOther),
          })}
        </Text>
      ) : null}
      {preview.preview.length > 0 ? (
        <Stack gap="1.5">
          <Text color="fg.muted" fontSize="xs" fontWeight="600" textTransform="uppercase">
            {t('cliptags.manager.import.preview')}
          </Text>
          <Scrollable
            borderColor="border.subtle"
            borderWidth="1px"
            label={t('cliptags.manager.import.preview')}
            maxH="48"
            rounded="md"
          >
            <Table.Root size="md">
              <Table.Body>
                {preview.preview.map((row, index) => (
                  <Table.Row key={`${row.content}:${row.type}:${index}`}>
                    <Table.Cell {...TABLE_CELL_PROPS}>{row.content}</Table.Cell>
                    <Table.Cell {...TABLE_CELL_PROPS} w="1">
                      <TagTypeBadge type={row.type} />
                    </Table.Cell>
                    <Table.Cell
                      {...TABLE_CELL_PROPS}
                      color="fg.muted"
                      fontVariantNumeric="tabular-nums"
                      textAlign="end"
                    >
                      {row.popularity === null ? '—' : formatCount(row.popularity)}
                    </Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Root>
          </Scrollable>
        </Stack>
      ) : null}
      <Field label={t('cliptags.manager.import.destination')}>
        <OptionSelect
          inDialog
          label={t('cliptags.manager.import.destination')}
          options={destinationOptions}
          value={destination.kind}
          onChange={handleKindChange}
        />
      </Field>
      {destination.kind === 'new_set' ? (
        <Field label={t('cliptags.manager.import.setName')}>
          <Input autoComplete="off" value={destination.newSetName} onChange={handleNewSetNameChange} />
        </Field>
      ) : null}
      {destination.kind === 'existing_set' ? (
        <Stack gap="3">
          <Field label={t('cliptags.manager.import.tagSet')}>
            {tagSets.length === 0 ? (
              <Text color="fg.muted" fontSize="md">
                {t('cliptags.manager.import.noTagSets')}
              </Text>
            ) : (
              <OptionSelect
                inDialog
                label={t('cliptags.manager.import.tagSet')}
                options={tagSetOptions}
                placeholder={t('cliptags.manager.import.selectTagSet')}
                value={destination.tagSetId}
                onChange={(tagSetId) => onChange({ tagSetId })}
              />
            )}
          </Field>
          <Field
            helpText={
              destination.mode === 'replace'
                ? t('cliptags.manager.import.modeHints.replace')
                : t('cliptags.manager.import.modeHints.merge')
            }
            label={t('cliptags.manager.import.importMode')}
          >
            <OptionSelect
              inDialog
              label={t('cliptags.manager.import.importMode')}
              options={modeOptions}
              value={destination.mode}
              onChange={handleModeChange}
            />
          </Field>
        </Stack>
      ) : null}
    </Stack>
  );
};

export const ResultStep = ({ result }: { result: ClipImportResult }) => {
  const { t } = useTranslation();
  const rows: { label: string; value: number }[] = [
    { label: t('cliptags.manager.import.newTags'), value: result.newTags },
    { label: t('cliptags.manager.import.existingMerged'), value: result.existingTagsMerged },
    { label: t('cliptags.manager.import.popularityUpdated'), value: result.popularityUpdated },
    { label: t('cliptags.manager.import.conflictsIgnored'), value: result.typeConflictsIgnored },
    { label: t('cliptags.manager.import.skippedRows'), value: result.skippedRows },
  ];

  return (
    <Stack gap="3">
      <Text fontSize="lg" fontWeight="600">
        {t('cliptags.manager.import.complete')}
      </Text>
      <Box as="dl" borderColor="border.subtle" borderTopWidth="1px">
        {rows.map((row) => (
          <HStack
            key={row.label}
            borderBottomWidth="1px"
            borderColor="border.subtle"
            gap="3"
            justify="space-between"
            py="2"
          >
            <Text as="dt" color="fg.muted" fontSize="md">
              {row.label}
            </Text>
            <Text as="dd" fontSize="md" fontVariantNumeric="tabular-nums" fontWeight="600">
              {formatCount(row.value)}
            </Text>
          </HStack>
        ))}
      </Box>
    </Stack>
  );
};

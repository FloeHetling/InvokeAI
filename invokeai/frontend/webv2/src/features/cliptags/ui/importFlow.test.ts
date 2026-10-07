import type { ClipImportPreview, ClipImportResult, ClipImportStage } from '@features/cliptags/core/types';

import { describe, expect, it } from 'vitest';

import {
  createMappingDraft,
  importReducer,
  INITIAL_IMPORT_STATE,
  isImportInFlight,
  NO_COLUMN,
  toImportDestination,
  type ImportAction,
  type ImportState,
} from './importFlow';

const STAGE: ClipImportStage = {
  columns: ['Type', 'Tag', 'Popularity'],
  detectedDelimiter: ';',
  detectedHeader: true,
  isSingleColumn: false,
  sampleRows: [['general', 'red_hair', '10']],
  sessionId: 'session-1',
};
const PREVIEW: ClipImportPreview = {
  preview: [],
  sessionId: 'session-1',
  summary: { invalidPopularityToUnknown: 0, rowsRead: 3, skippedRows: 0, unknownTypesToOther: 0, validRows: 3 },
};
const RESULT: ClipImportResult = {
  existingTagsMerged: 0,
  newTags: 3,
  popularityUpdated: 0,
  skippedRows: 0,
  typeConflictsIgnored: 0,
};

const run = (...actions: ImportAction[]): ImportState =>
  actions.reduce<ImportState>((state, action) => importReducer(state, action), INITIAL_IMPORT_STATE);

describe('createMappingDraft', () => {
  it('follows the detection and picks the columns a header names', () => {
    expect(createMappingDraft(STAGE)).toEqual({
      delimiter: ';',
      firstRowContainsColumnNames: true,
      popularityColumn: 2,
      tagColumn: 1,
      typeColumn: 0,
    });
  });

  it('reads a single column as tag text only and leaves unnamed optional columns out', () => {
    expect(createMappingDraft({ ...STAGE, columns: ['red_hair'], isSingleColumn: true })).toMatchObject({
      popularityColumn: NO_COLUMN,
      tagColumn: 0,
      typeColumn: NO_COLUMN,
    });
    expect(createMappingDraft({ ...STAGE, columns: ['a', 'b'] })).toMatchObject({
      popularityColumn: NO_COLUMN,
      tagColumn: 0,
      typeColumn: NO_COLUMN,
    });
  });
});

describe('mapping and destination', () => {
  it('needs a name for a new set and a set for an existing one', () => {
    const base = { kind: 'uncategorized' as const, mode: 'merge' as const, newSetName: '', tagSetId: '' };

    expect(toImportDestination(base)).toEqual({ type: 'uncategorized' });
    expect(toImportDestination({ ...base, kind: 'new_set', newSetName: '  ' })).toBeNull();
    expect(toImportDestination({ ...base, kind: 'new_set', newSetName: ' Anime ' })).toEqual({
      name: 'Anime',
      type: 'new_set',
    });
    expect(toImportDestination({ ...base, kind: 'existing_set' })).toBeNull();
    expect(toImportDestination({ ...base, kind: 'existing_set', mode: 'replace', tagSetId: 's1' })).toEqual({
      mode: 'replace',
      tagSetId: 's1',
      type: 'existing_set',
    });
  });
});

describe('importReducer', () => {
  it('walks the happy path from file to result', () => {
    const staged = run({ type: 'stageStarted' }, { stage: STAGE, type: 'stageSucceeded' });

    expect(staged.step).toBe('mapping');

    const previewed = importReducer(importReducer(staged, { type: 'prepareStarted' }), {
      preview: PREVIEW,
      type: 'prepareSucceeded',
    });

    expect(previewed.step).toBe('preview');

    const committing = importReducer(
      importReducer(previewed, { changes: { kind: 'new_set', newSetName: 'Anime' }, type: 'destinationChanged' }),
      { type: 'commitStarted' }
    );

    expect(committing.step).toBe('committing');
    expect(isImportInFlight(committing)).toBe(true);

    const done = importReducer(committing, { result: RESULT, type: 'commitSucceeded' });

    expect(done).toEqual({ result: RESULT, step: 'result' });
  });

  it('returns to the step that failed, keeping the staged file and the user input', () => {
    const mapping = run({ type: 'stageStarted' }, { stage: STAGE, type: 'stageSucceeded' });
    const prepared = importReducer(mapping, { type: 'prepareStarted' });
    const failedPrepare = importReducer(prepared, { message: 'bad column', type: 'prepareFailed' });

    expect(failedPrepare).toMatchObject({ error: 'bad column', stage: STAGE, step: 'mapping' });

    const previewed = importReducer(importReducer(failedPrepare, { type: 'prepareStarted' }), {
      preview: PREVIEW,
      type: 'prepareSucceeded',
    });
    const typed = importReducer(previewed, {
      changes: { kind: 'new_set', newSetName: 'Anime' },
      type: 'destinationChanged',
    });
    const failedCommit = importReducer(importReducer(typed, { type: 'commitStarted' }), {
      message: 'busy',
      type: 'commitFailed',
    });

    expect(failedCommit).toMatchObject({
      destination: { kind: 'new_set', newSetName: 'Anime' },
      error: 'busy',
      step: 'preview',
    });
    expect(failedCommit).toMatchObject({ stage: STAGE });
  });

  it('cannot be reset while a request is in flight, so the session is never orphaned mid-request', () => {
    const preparing = run(
      { type: 'stageStarted' },
      { stage: STAGE, type: 'stageSucceeded' },
      { type: 'prepareStarted' }
    );

    expect(importReducer(preparing, { type: 'reset' })).toBe(preparing);
    expect(importReducer(importReducer(preparing, { message: 'x', type: 'prepareFailed' }), { type: 'reset' })).toBe(
      INITIAL_IMPORT_STATE
    );
  });
});

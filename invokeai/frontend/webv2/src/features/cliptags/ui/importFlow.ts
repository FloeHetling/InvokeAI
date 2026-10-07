import type {
  ClipImportDelimiter,
  ClipImportDestination,
  ClipImportMapping,
  ClipImportPreview,
  ClipImportResult,
  ClipImportStage,
} from '@features/cliptags/core/types';

/** An optional column left out of the import. */
export const NO_COLUMN = -1;

export const IMPORT_DELIMITERS: readonly ClipImportDelimiter[] = [',', ';', '\t', '|'];

/** The mapping as the user edits it: every column is an index, with `NO_COLUMN` for an ignored optional one. */
export interface ImportMappingDraft {
  delimiter: ClipImportDelimiter;
  firstRowContainsColumnNames: boolean;
  popularityColumn: number;
  tagColumn: number;
  typeColumn: number;
}

export interface ImportDestinationDraft {
  kind: ClipImportDestination['type'];
  mode: 'merge' | 'replace';
  newSetName: string;
  tagSetId: string;
}

export type ImportState =
  | { error: string | null; step: 'select' }
  | { step: 'staging' }
  | { error: string | null; mapping: ImportMappingDraft; stage: ClipImportStage; step: 'mapping' }
  | { mapping: ImportMappingDraft; stage: ClipImportStage; step: 'preparing' }
  | {
      destination: ImportDestinationDraft;
      error: string | null;
      mapping: ImportMappingDraft;
      preview: ClipImportPreview;
      stage: ClipImportStage;
      step: 'preview';
    }
  | {
      destination: ImportDestinationDraft;
      mapping: ImportMappingDraft;
      preview: ClipImportPreview;
      stage: ClipImportStage;
      step: 'committing';
    }
  | { result: ClipImportResult; step: 'result' };

export type ImportAction =
  | { type: 'stageStarted' }
  | { stage: ClipImportStage; type: 'stageSucceeded' }
  | { message: string; type: 'stageFailed' }
  | { changes: Partial<ImportMappingDraft>; type: 'mappingChanged' }
  | { type: 'prepareStarted' }
  | { preview: ClipImportPreview; type: 'prepareSucceeded' }
  | { message: string; type: 'prepareFailed' }
  | { type: 'mappingReopened' }
  | { changes: Partial<ImportDestinationDraft>; type: 'destinationChanged' }
  | { type: 'commitStarted' }
  | { result: ClipImportResult; type: 'commitSucceeded' }
  | { message: string; type: 'commitFailed' }
  | { type: 'reset' };

export const INITIAL_IMPORT_STATE: ImportState = { error: null, step: 'select' };

const columnIndex = (columns: readonly string[], name: string): number =>
  columns.findIndex((column) => column.trim().toLowerCase() === name);

/** Starts from the server's detection, and picks the columns a header names `tag`, `popularity` and `type`. */
export const createMappingDraft = (stage: ClipImportStage): ImportMappingDraft => ({
  delimiter: stage.detectedDelimiter,
  firstRowContainsColumnNames: stage.detectedHeader,
  popularityColumn: stage.isSingleColumn ? NO_COLUMN : columnIndex(stage.columns, 'popularity'),
  tagColumn: Math.max(0, columnIndex(stage.columns, 'tag')),
  typeColumn: stage.isSingleColumn ? NO_COLUMN : columnIndex(stage.columns, 'type'),
});

/** Two fields cannot read one column; the server refuses it too. */
export const hasDuplicateColumns = (mapping: ImportMappingDraft): boolean => {
  const mapped = [mapping.tagColumn, mapping.popularityColumn, mapping.typeColumn].filter(
    (column) => column !== NO_COLUMN
  );

  return new Set(mapped).size !== mapped.length;
};

export const toImportMapping = (mapping: ImportMappingDraft): ClipImportMapping => ({
  delimiter: mapping.delimiter,
  firstRowContainsColumnNames: mapping.firstRowContainsColumnNames,
  popularityColumn: mapping.popularityColumn === NO_COLUMN ? null : mapping.popularityColumn,
  tagColumn: mapping.tagColumn,
  typeColumn: mapping.typeColumn === NO_COLUMN ? null : mapping.typeColumn,
});

/** Null while the chosen destination is incomplete. */
export const toImportDestination = (draft: ImportDestinationDraft): ClipImportDestination | null => {
  switch (draft.kind) {
    case 'uncategorized':
      return { type: 'uncategorized' };
    case 'new_set': {
      const name = draft.newSetName.trim();

      return name ? { name, type: 'new_set' } : null;
    }
    case 'existing_set':
      return draft.tagSetId ? { mode: draft.mode, tagSetId: draft.tagSetId, type: 'existing_set' } : null;
  }
};

export const isImportInFlight = (state: ImportState): boolean =>
  state.step === 'staging' || state.step === 'preparing' || state.step === 'committing';

const INITIAL_DESTINATION: ImportDestinationDraft = {
  kind: 'uncategorized',
  mode: 'merge',
  newSetName: '',
  tagSetId: '',
};

/** Transitions out of a step that does not expect the action are ignored, so a late answer cannot corrupt the flow. */
export const importReducer = (state: ImportState, action: ImportAction): ImportState => {
  switch (action.type) {
    case 'reset':
      return state.step === 'staging' || state.step === 'preparing' || state.step === 'committing'
        ? state
        : INITIAL_IMPORT_STATE;
    case 'stageStarted':
      return state.step === 'select' ? { step: 'staging' } : state;
    case 'stageSucceeded':
      return state.step === 'staging'
        ? { error: null, mapping: createMappingDraft(action.stage), stage: action.stage, step: 'mapping' }
        : state;
    case 'stageFailed':
      return state.step === 'staging' ? { error: action.message, step: 'select' } : state;
    case 'mappingChanged':
      return state.step === 'mapping'
        ? { ...state, error: null, mapping: { ...state.mapping, ...action.changes } }
        : state;
    case 'prepareStarted':
      return state.step === 'mapping' && !hasDuplicateColumns(state.mapping)
        ? { mapping: state.mapping, stage: state.stage, step: 'preparing' }
        : state;
    case 'prepareSucceeded':
      return state.step === 'preparing'
        ? {
            destination: INITIAL_DESTINATION,
            error: null,
            mapping: state.mapping,
            preview: action.preview,
            stage: state.stage,
            step: 'preview',
          }
        : state;
    case 'prepareFailed':
      return state.step === 'preparing'
        ? { error: action.message, mapping: state.mapping, stage: state.stage, step: 'mapping' }
        : state;
    case 'mappingReopened':
      return state.step === 'preview'
        ? { error: null, mapping: state.mapping, stage: state.stage, step: 'mapping' }
        : state;
    case 'destinationChanged':
      return state.step === 'preview'
        ? { ...state, destination: { ...state.destination, ...action.changes }, error: null }
        : state;
    case 'commitStarted':
      return state.step === 'preview' && toImportDestination(state.destination) !== null
        ? {
            destination: state.destination,
            mapping: state.mapping,
            preview: state.preview,
            stage: state.stage,
            step: 'committing',
          }
        : state;
    case 'commitSucceeded':
      return state.step === 'committing' ? { result: action.result, step: 'result' } : state;
    case 'commitFailed':
      return state.step === 'committing'
        ? {
            destination: state.destination,
            error: action.message,
            mapping: state.mapping,
            preview: state.preview,
            stage: state.stage,
            step: 'preview',
          }
        : state;
  }
};

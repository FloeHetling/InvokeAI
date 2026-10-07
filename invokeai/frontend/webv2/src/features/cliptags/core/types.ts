export type ClipTagType = 'general' | 'artist' | 'copyright' | 'character' | 'meta' | 'other';

export const CLIP_TAG_TYPES: readonly ClipTagType[] = ['general', 'artist', 'copyright', 'character', 'meta', 'other'];

/** Chakra `colorPalette` for each tag type, so every surface colors a type the same way. */
export const CLIP_TAG_TYPE_PALETTE: Record<ClipTagType, string> = {
  artist: 'red',
  character: 'green',
  copyright: 'purple',
  general: 'blue',
  meta: 'yellow',
  other: 'gray',
};

/** Categories the prompt picker can browse; every other type is reachable by search only. */
export type ClipTagCategory = 'artist' | 'character' | 'other';

export const CLIP_TAG_CATEGORIES: readonly ClipTagCategory[] = ['artist', 'character', 'other'];

export type ClipTagStatusReason = 'fts5_unavailable' | 'database_incompatible' | 'database_error';

/** Unavailable means the sidecar database could not be opened; the reason is absent when no service is running. */
export interface ClipTagStatus {
  available: boolean;
  reason: ClipTagStatusReason | null;
}

export interface ClipTag {
  id: string;
  content: string;
  popularity: number | null;
  type: ClipTagType;
}

/** A search hit: the tag plus its text as the model's syntax profile renders it. */
export interface ClipTagCandidate extends ClipTag {
  renderedContent: string;
}

export interface ClipTagDetail extends ClipTag {
  tagSetIds: string[];
}

export interface ClipTagSet {
  id: string;
  name: string;
  tagCount: number;
  modelCount: number;
}

export interface ClipSyntaxProfileOptions {
  appendTypeParentheses: boolean;
  escapeColons: boolean;
  escapeParentheses: boolean;
  prefixArtistWithBy: boolean;
  spacesToUnderscores: boolean;
}

export interface ClipSyntaxProfile extends ClipSyntaxProfileOptions {
  id: string;
  name: string;
}

export interface ClipModelConfig {
  modelId: string;
  syntaxProfileId: string | null;
  tagSetIds: string[];
}

/** Tag set and uncategorized exclude each other; the backend refuses both. */
export interface ClipTagFilter {
  search?: string;
  tagSetId?: string;
  tagType?: ClipTagType;
  uncategorized?: boolean;
}

export interface ClipTagPage {
  items: ClipTag[];
  nextCursor: string | null;
  total: number;
}

export interface ClipTagUpdate {
  content?: string;
  /** Null clears the popularity back to unknown. */
  popularity?: number | null;
  tagSetIds?: string[];
  type?: ClipTagType;
}

export interface ClipTagMutationResult {
  merged: boolean;
  tag: ClipTagDetail;
}

export type ClipTagBulkSelection = { mode: 'ids'; ids: string[] } | { mode: 'filter'; filter: ClipTagFilter };

export type ClipTagBulkOperation =
  | { type: 'add_to_set'; tagSetId: string }
  | { type: 'remove_from_set'; tagSetId: string }
  | { type: 'set_type'; tagType: ClipTagType }
  | { type: 'delete' };

export interface ClipTagBulkResult {
  affectedCount: number;
  mergedCount: number;
  selectedCount: number;
}

export type ClipImportDelimiter = ',' | ';' | '\t' | '|';

export interface ClipImportStage {
  columns: string[];
  detectedDelimiter: ClipImportDelimiter;
  detectedHeader: boolean;
  isSingleColumn: boolean;
  sampleRows: string[][];
  sessionId: string;
}

export interface ClipImportMapping {
  delimiter: ClipImportDelimiter;
  firstRowContainsColumnNames: boolean;
  popularityColumn: number | null;
  tagColumn: number;
  typeColumn: number | null;
}

export interface ClipImportPreview {
  preview: { content: string; popularity: number | null; type: ClipTagType }[];
  sessionId: string;
  summary: {
    invalidPopularityToUnknown: number;
    rowsRead: number;
    skippedRows: number;
    unknownTypesToOther: number;
    validRows: number;
  };
}

export type ClipImportDestination =
  | { type: 'uncategorized' }
  | { type: 'new_set'; name: string }
  | { type: 'existing_set'; tagSetId: string; mode: 'merge' | 'replace' };

export interface ClipImportResult {
  existingTagsMerged: number;
  newTags: number;
  popularityUpdated: number;
  skippedRows: number;
  typeConflictsIgnored: number;
}

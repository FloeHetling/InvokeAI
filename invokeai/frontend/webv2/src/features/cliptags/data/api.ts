import type {
  ClipImportDelimiter,
  ClipImportDestination,
  ClipImportMapping,
  ClipImportPreview,
  ClipImportResult,
  ClipImportStage,
  ClipModelConfig,
  ClipSyntaxProfile,
  ClipSyntaxProfileOptions,
  ClipTag,
  ClipTagBulkOperation,
  ClipTagBulkResult,
  ClipTagBulkSelection,
  ClipTagCandidate,
  ClipTagCategory,
  ClipTagDetail,
  ClipTagFilter,
  ClipTagMutationResult,
  ClipTagPage,
  ClipTagSet,
  ClipTagStatus,
  ClipTagType,
  ClipTagUpdate,
} from '@features/cliptags/core/types';

import { apiFetch, apiFetchJson } from '@platform/transport/http';

const BASE = '/api/v1/clip_tag_autocomplete';

interface TagDTO {
  id: string;
  canonical_content: string;
  popularity?: number | null;
  tag_type: ClipTagType;
}

interface CandidateDTO extends TagDTO {
  rendered_content: string;
}

interface TagDetailDTO extends TagDTO {
  tag_set_ids?: string[];
}

interface TagSetDTO {
  id: string;
  name: string;
  tag_count?: number;
  model_count?: number;
}

interface SyntaxProfileDTO {
  id: string;
  name: string;
  spaces_to_underscores?: boolean;
  escape_parentheses?: boolean;
  escape_colons?: boolean;
  append_type_parentheses?: boolean;
  prefix_artist_with_by?: boolean;
}

interface ModelConfigDTO {
  model_id: string;
  syntax_profile_id?: string | null;
  tag_set_ids?: string[];
}

const mapTag = (dto: TagDTO): ClipTag => ({
  content: dto.canonical_content,
  id: dto.id,
  popularity: dto.popularity ?? null,
  type: dto.tag_type,
});

const mapCandidate = (dto: CandidateDTO): ClipTagCandidate => ({
  ...mapTag(dto),
  renderedContent: dto.rendered_content,
});

const mapTagDetail = (dto: TagDetailDTO): ClipTagDetail => ({ ...mapTag(dto), tagSetIds: dto.tag_set_ids ?? [] });

const mapTagSet = (dto: TagSetDTO): ClipTagSet => ({
  id: dto.id,
  modelCount: dto.model_count ?? 0,
  name: dto.name,
  tagCount: dto.tag_count ?? 0,
});

const mapSyntaxProfile = (dto: SyntaxProfileDTO): ClipSyntaxProfile => ({
  appendTypeParentheses: dto.append_type_parentheses ?? false,
  escapeColons: dto.escape_colons ?? true,
  escapeParentheses: dto.escape_parentheses ?? true,
  id: dto.id,
  name: dto.name,
  prefixArtistWithBy: dto.prefix_artist_with_by ?? false,
  spacesToUnderscores: dto.spaces_to_underscores ?? true,
});

const mapModelConfig = (dto: ModelConfigDTO): ClipModelConfig => ({
  modelId: dto.model_id,
  syntaxProfileId: dto.syntax_profile_id ?? null,
  tagSetIds: dto.tag_set_ids ?? [],
});

const toSyntaxProfileBody = (options: Partial<ClipSyntaxProfileOptions>) => ({
  append_type_parentheses: options.appendTypeParentheses,
  escape_colons: options.escapeColons,
  escape_parentheses: options.escapeParentheses,
  prefix_artist_with_by: options.prefixArtistWithBy,
  spaces_to_underscores: options.spacesToUnderscores,
});

const jsonRequest = (method: string, body: unknown, signal?: AbortSignal): RequestInit => ({
  body: JSON.stringify(body),
  method,
  signal,
});

const toQuery = (entries: Record<string, boolean | number | string | undefined>): string => {
  const params = new URLSearchParams();

  for (const [key, value] of Object.entries(entries)) {
    if (value !== undefined && value !== '') {
      params.set(key, String(value));
    }
  }

  const query = params.toString();

  return query ? `?${query}` : '';
};

const toFilterBody = (filter: ClipTagFilter) => ({
  q: filter.search || null,
  tag_set_id: filter.tagSetId ?? null,
  tag_type: filter.tagType ?? null,
  uncategorized: filter.uncategorized ?? false,
});

export const getClipTagStatus = async (signal?: AbortSignal): Promise<ClipTagStatus> => {
  const dto = await apiFetchJson<{ available: boolean; reason?: ClipTagStatus['reason'] }>(`${BASE}/status`, {
    signal,
  });

  return { available: dto.available, reason: dto.reason ?? null };
};

export interface SearchClipTagsParams {
  /** Narrows to one category; pages are then requested with `offset` and `limit`. */
  category?: ClipTagCategory;
  limit?: number;
  modelId?: string | null;
  offset?: number;
  query: string;
}

/** Candidates arrive already rendered for the model's syntax profile. */
export const searchClipTags = async (
  { category, limit, modelId, offset, query }: SearchClipTagsParams,
  signal?: AbortSignal
): Promise<ClipTagCandidate[]> => {
  const dtos = await apiFetchJson<CandidateDTO[]>(
    `${BASE}/autocomplete${toQuery({
      limit,
      model_id: modelId ?? undefined,
      offset,
      q: query,
      tag_filter: category,
    })}`,
    { signal }
  );

  return dtos.map(mapCandidate);
};

export const listClipTags = async (
  filter: ClipTagFilter,
  { cursor, limit }: { cursor?: string | null; limit?: number } = {},
  signal?: AbortSignal
): Promise<ClipTagPage> => {
  const dto = await apiFetchJson<{ items: TagDTO[]; next_cursor?: string | null; total_count: number }>(
    `${BASE}/tags${toQuery({
      cursor: cursor ?? undefined,
      limit,
      q: filter.search,
      tag_set_id: filter.tagSetId,
      tag_type: filter.tagType,
      uncategorized: filter.uncategorized ? true : undefined,
    })}`,
    { signal }
  );

  return { items: dto.items.map(mapTag), nextCursor: dto.next_cursor ?? null, total: dto.total_count };
};

export const getClipTag = async (id: string, signal?: AbortSignal): Promise<ClipTagDetail> =>
  mapTagDetail(await apiFetchJson<TagDetailDTO>(`${BASE}/tags/i/${encodeURIComponent(id)}`, { signal }));

export const updateClipTag = async (
  id: string,
  update: ClipTagUpdate,
  signal?: AbortSignal
): Promise<ClipTagMutationResult> => {
  const dto = await apiFetchJson<{ merged?: boolean; tag: TagDetailDTO }>(
    `${BASE}/tags/i/${encodeURIComponent(id)}`,
    jsonRequest(
      'PATCH',
      {
        canonical_content: update.content,
        popularity: update.popularity,
        tag_set_ids: update.tagSetIds,
        tag_type: update.type,
      },
      signal
    )
  );

  return { merged: dto.merged ?? false, tag: mapTagDetail(dto.tag) };
};

export const deleteClipTag = async (id: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${BASE}/tags/i/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
};

const toBulkSelectionBody = (selection: ClipTagBulkSelection) =>
  selection.mode === 'ids'
    ? { ids: selection.ids, mode: 'ids' as const }
    : { filter: toFilterBody(selection.filter), mode: 'filter' as const };

const toBulkOperationBody = (operation: ClipTagBulkOperation) => {
  switch (operation.type) {
    case 'add_to_set':
    case 'remove_from_set':
      return { tag_set_id: operation.tagSetId, type: operation.type };
    case 'set_type':
      return { tag_type: operation.tagType, type: operation.type };
    case 'delete':
      return { type: operation.type };
  }
};

export const bulkMutateClipTags = async (
  selection: ClipTagBulkSelection,
  operation: ClipTagBulkOperation,
  signal?: AbortSignal
): Promise<ClipTagBulkResult> => {
  const dto = await apiFetchJson<{ affected_count: number; merged_count?: number; selected_count: number }>(
    `${BASE}/tags/bulk`,
    jsonRequest(
      'POST',
      { operation: toBulkOperationBody(operation), selection: toBulkSelectionBody(selection) },
      signal
    )
  );

  return { affectedCount: dto.affected_count, mergedCount: dto.merged_count ?? 0, selectedCount: dto.selected_count };
};

export const listClipTagSets = async (signal?: AbortSignal): Promise<ClipTagSet[]> =>
  (await apiFetchJson<TagSetDTO[]>(`${BASE}/tag_sets`, { signal })).map(mapTagSet);

export const createClipTagSet = async (name: string, signal?: AbortSignal): Promise<ClipTagSet> =>
  mapTagSet(await apiFetchJson<TagSetDTO>(`${BASE}/tag_sets`, jsonRequest('POST', { name }, signal)));

export const renameClipTagSet = async (id: string, name: string, signal?: AbortSignal): Promise<ClipTagSet> =>
  mapTagSet(
    await apiFetchJson<TagSetDTO>(
      `${BASE}/tag_sets/i/${encodeURIComponent(id)}`,
      jsonRequest('PATCH', { name }, signal)
    )
  );

export const deleteClipTagSet = async (id: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${BASE}/tag_sets/i/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
};

export const listClipSyntaxProfiles = async (signal?: AbortSignal): Promise<ClipSyntaxProfile[]> =>
  (await apiFetchJson<SyntaxProfileDTO[]>(`${BASE}/syntax_profiles`, { signal })).map(mapSyntaxProfile);

export const createClipSyntaxProfile = async (
  name: string,
  options: Partial<ClipSyntaxProfileOptions> = {},
  signal?: AbortSignal
): Promise<ClipSyntaxProfile> =>
  mapSyntaxProfile(
    await apiFetchJson<SyntaxProfileDTO>(
      `${BASE}/syntax_profiles`,
      jsonRequest('POST', { name, ...toSyntaxProfileBody(options) }, signal)
    )
  );

export const updateClipSyntaxProfile = async (
  id: string,
  changes: Partial<ClipSyntaxProfileOptions> & { name?: string },
  signal?: AbortSignal
): Promise<ClipSyntaxProfile> =>
  mapSyntaxProfile(
    await apiFetchJson<SyntaxProfileDTO>(
      `${BASE}/syntax_profiles/i/${encodeURIComponent(id)}`,
      jsonRequest('PATCH', { name: changes.name, ...toSyntaxProfileBody(changes) }, signal)
    )
  );

export const deleteClipSyntaxProfile = async (id: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${BASE}/syntax_profiles/i/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
};

export const getClipModelConfig = async (modelId: string, signal?: AbortSignal): Promise<ClipModelConfig> =>
  mapModelConfig(
    await apiFetchJson<ModelConfigDTO>(`${BASE}/models/i/${encodeURIComponent(modelId)}/config`, { signal })
  );

export const setClipModelConfig = async (
  modelId: string,
  config: { syntaxProfileId: string | null; tagSetIds: string[] },
  signal?: AbortSignal
): Promise<ClipModelConfig> =>
  mapModelConfig(
    await apiFetchJson<ModelConfigDTO>(
      `${BASE}/models/i/${encodeURIComponent(modelId)}/config`,
      jsonRequest('PUT', { syntax_profile_id: config.syntaxProfileId, tag_set_ids: config.tagSetIds }, signal)
    )
  );

export const downloadClipSampleCsv = async (signal?: AbortSignal): Promise<Blob> =>
  (await apiFetch(`${BASE}/imports/sample`, { signal })).blob();

export const stageClipImport = async (file: Blob, signal?: AbortSignal): Promise<ClipImportStage> => {
  const body = new FormData();

  body.append('file', file);

  const dto = await apiFetchJson<{
    columns: string[];
    detected_delimiter: ClipImportDelimiter;
    detected_header: boolean;
    is_single_column: boolean;
    sample_rows: string[][];
    session_id: string;
  }>(`${BASE}/imports/stage`, { body, method: 'POST', signal });

  return {
    columns: dto.columns,
    detectedDelimiter: dto.detected_delimiter,
    detectedHeader: dto.detected_header,
    isSingleColumn: dto.is_single_column,
    sampleRows: dto.sample_rows,
    sessionId: dto.session_id,
  };
};

export const prepareClipImport = async (
  sessionId: string,
  mapping: ClipImportMapping,
  signal?: AbortSignal
): Promise<ClipImportPreview> => {
  const dto = await apiFetchJson<{
    preview: { canonical_content: string; popularity?: number | null; tag_type: ClipTagType }[];
    session_id: string;
    summary: {
      invalid_popularity_to_unknown: number;
      rows_read: number;
      skipped_rows: number;
      unknown_types_to_other: number;
      valid_rows: number;
    };
  }>(
    `${BASE}/imports/i/${encodeURIComponent(sessionId)}/prepare`,
    jsonRequest(
      'POST',
      {
        delimiter: mapping.delimiter,
        first_row_contains_column_names: mapping.firstRowContainsColumnNames,
        popularity_column: mapping.popularityColumn,
        tag_column: mapping.tagColumn,
        type_column: mapping.typeColumn,
      },
      signal
    )
  );

  return {
    preview: dto.preview.map((row) => ({
      content: row.canonical_content,
      popularity: row.popularity ?? null,
      type: row.tag_type,
    })),
    sessionId: dto.session_id,
    summary: {
      invalidPopularityToUnknown: dto.summary.invalid_popularity_to_unknown,
      rowsRead: dto.summary.rows_read,
      skippedRows: dto.summary.skipped_rows,
      unknownTypesToOther: dto.summary.unknown_types_to_other,
      validRows: dto.summary.valid_rows,
    },
  };
};

const toDestinationBody = (destination: ClipImportDestination) =>
  destination.type === 'existing_set'
    ? { mode: destination.mode, tag_set_id: destination.tagSetId, type: destination.type }
    : destination;

export const commitClipImport = async (
  sessionId: string,
  destination: ClipImportDestination,
  signal?: AbortSignal
): Promise<ClipImportResult> => {
  const dto = await apiFetchJson<{
    existing_tags_merged: number;
    new_tags: number;
    popularity_updated: number;
    skipped_rows: number;
    type_conflicts_ignored: number;
  }>(
    `${BASE}/imports/i/${encodeURIComponent(sessionId)}/commit`,
    jsonRequest('POST', { destination: toDestinationBody(destination) }, signal)
  );

  return {
    existingTagsMerged: dto.existing_tags_merged,
    newTags: dto.new_tags,
    popularityUpdated: dto.popularity_updated,
    skippedRows: dto.skipped_rows,
    typeConflictsIgnored: dto.type_conflicts_ignored,
  };
};

export const cancelClipImport = async (sessionId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${BASE}/imports/i/${encodeURIComponent(sessionId)}`, { method: 'DELETE', signal });
};

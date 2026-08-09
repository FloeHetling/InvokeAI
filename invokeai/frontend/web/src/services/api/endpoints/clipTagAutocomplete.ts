import type { S } from 'services/api/types';

import { api, buildV1Url, LIST_TAG } from '..';

type CtaAutocompleteCandidate = S['CtaAutocompleteCandidate'];
export type CtaAutocompleteFilter = 'artist' | 'character' | 'other';
export type CtaAutocompleteArgs = {
  q: string;
  model_id?: string;
  tag_filter?: CtaAutocompleteFilter;
  offset?: number;
  limit?: number;
};
type CtaStatus = S['CtaStatus'];
type CtaTagDetailDTO = S['CtaTagDetailDTO'];
type CtaTagMutationResult = S['CtaTagMutationResult'];
type CtaTagPage = S['CtaTagPage'];
type CtaTagUpdate = S['CtaTagUpdate'];
type CtaTagSetDTO = S['CtaTagSetDTO'];
type CtaSyntaxProfileDTO = S['CtaSyntaxProfileDTO'];
type CtaModelConfigDTO = S['CtaModelConfigDTO'];
type CtaModelConfigUpdate = S['CtaModelConfigUpdate'];
type CtaBulkTagRequest = S['CtaBulkTagRequest'];
type CtaBulkResult = S['CtaBulkResult'];
type CtaImportMapping = S['CtaImportMapping'];
type CtaImportStageDTO = S['CtaImportStageDTO'];
type CtaImportPreviewDTO = S['CtaImportPreviewDTO'];
type CtaImportCommitRequest = S['CtaImportCommitRequest'];
type CtaImportResultDTO = S['CtaImportResultDTO'];

const buildCtaUrl = (path: string = '') => buildV1Url(`clip_tag_autocomplete/${path}`);

const clipTagAutocompleteApi = api.injectEndpoints({
  endpoints: (build) => ({
    getCtaStatus: build.query<CtaStatus, void>({
      query: () => ({ url: buildCtaUrl('status') }),
      providesTags: ['CtaStatus'],
    }),

    autocompleteCta: build.query<CtaAutocompleteCandidate[], CtaAutocompleteArgs>({
      query: ({ q, model_id, tag_filter, offset, limit }) => ({
        url: buildCtaUrl('autocomplete'),
        params: { q, model_id, tag_filter, offset, limit },
      }),
      keepUnusedDataFor: 5,
    }),

    listCtaTags: build.query<
      CtaTagPage,
      {
        q?: string;
        tag_type?: string;
        tag_set_id?: string;
        uncategorized?: boolean;
        cursor?: string;
        limit?: number;
      }
    >({
      query: (params) => ({ url: buildCtaUrl('tags'), params }),
      providesTags: [{ type: 'CtaTags', id: LIST_TAG }],
    }),

    getCtaTag: build.query<CtaTagDetailDTO, string>({
      query: (tagId) => ({ url: buildCtaUrl(`tags/i/${tagId}`) }),
      providesTags: (_result, _error, tagId) => [{ type: 'CtaTags', id: tagId }],
    }),

    updateCtaTag: build.mutation<CtaTagMutationResult, { id: string; body: CtaTagUpdate }>({
      query: ({ id, body }) => ({
        url: buildCtaUrl(`tags/i/${id}`),
        method: 'PATCH',
        body,
      }),
      invalidatesTags: (_result, _error, { id }) => [
        { type: 'CtaTags', id },
        { type: 'CtaTags', id: LIST_TAG },
        { type: 'CtaTagSets', id: LIST_TAG },
      ],
    }),

    deleteCtaTag: build.mutation<void, string>({
      query: (id) => ({
        url: buildCtaUrl(`tags/i/${id}`),
        method: 'DELETE',
      }),
      invalidatesTags: [
        { type: 'CtaTags', id: LIST_TAG },
        { type: 'CtaTagSets', id: LIST_TAG },
      ],
    }),

    bulkMutateCtaTags: build.mutation<CtaBulkResult, CtaBulkTagRequest>({
      query: (body) => ({
        url: buildCtaUrl('tags/bulk'),
        method: 'POST',
        body,
      }),
      invalidatesTags: [
        { type: 'CtaTags', id: LIST_TAG },
        { type: 'CtaTagSets', id: LIST_TAG },
      ],
    }),

    listCtaTagSets: build.query<CtaTagSetDTO[], void>({
      query: () => ({ url: buildCtaUrl('tag_sets') }),
      providesTags: [{ type: 'CtaTagSets', id: LIST_TAG }],
    }),

    createCtaTagSet: build.mutation<CtaTagSetDTO, { name: string }>({
      query: (body) => ({
        url: buildCtaUrl('tag_sets'),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'CtaTagSets', id: LIST_TAG }],
    }),

    updateCtaTagSet: build.mutation<CtaTagSetDTO, { id: string; name: string }>({
      query: ({ id, ...body }) => ({
        url: buildCtaUrl(`tag_sets/i/${id}`),
        method: 'PATCH',
        body,
      }),
      invalidatesTags: [{ type: 'CtaTagSets', id: LIST_TAG }],
    }),

    deleteCtaTagSet: build.mutation<void, string>({
      query: (id) => ({
        url: buildCtaUrl(`tag_sets/i/${id}`),
        method: 'DELETE',
      }),
      invalidatesTags: [{ type: 'CtaTagSets', id: LIST_TAG }, { type: 'CtaTags', id: LIST_TAG }, 'CtaModelConfig'],
    }),

    listCtaSyntaxProfiles: build.query<CtaSyntaxProfileDTO[], void>({
      query: () => ({ url: buildCtaUrl('syntax_profiles') }),
      providesTags: [{ type: 'CtaSyntaxProfiles', id: LIST_TAG }],
    }),

    createCtaSyntaxProfile: build.mutation<CtaSyntaxProfileDTO, { name: string }>({
      query: (body) => ({
        url: buildCtaUrl('syntax_profiles'),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'CtaSyntaxProfiles', id: LIST_TAG }],
    }),

    updateCtaSyntaxProfile: build.mutation<
      CtaSyntaxProfileDTO,
      { id: string } & Partial<
        Pick<
          CtaSyntaxProfileDTO,
          | 'name'
          | 'spaces_to_underscores'
          | 'escape_parentheses'
          | 'escape_colons'
          | 'append_type_parentheses'
          | 'prefix_artist_with_by'
        >
      >
    >({
      query: ({ id, ...body }) => ({
        url: buildCtaUrl(`syntax_profiles/i/${id}`),
        method: 'PATCH',
        body,
      }),
      invalidatesTags: [{ type: 'CtaSyntaxProfiles', id: LIST_TAG }],
    }),

    deleteCtaSyntaxProfile: build.mutation<void, string>({
      query: (id) => ({
        url: buildCtaUrl(`syntax_profiles/i/${id}`),
        method: 'DELETE',
      }),
      invalidatesTags: [{ type: 'CtaSyntaxProfiles', id: LIST_TAG }, 'CtaModelConfig'],
    }),

    getCtaModelConfig: build.query<CtaModelConfigDTO, string>({
      query: (modelId) => ({ url: buildCtaUrl(`models/i/${modelId}/config`) }),
      providesTags: (_result, _error, modelId) => [{ type: 'CtaModelConfig', id: modelId }],
    }),

    setCtaModelConfig: build.mutation<CtaModelConfigDTO, { modelId: string; body: CtaModelConfigUpdate }>({
      query: ({ modelId, body }) => ({
        url: buildCtaUrl(`models/i/${modelId}/config`),
        method: 'PUT',
        body,
      }),
      invalidatesTags: (_result, _error, { modelId }) => [{ type: 'CtaModelConfig', id: modelId }],
    }),

    stageCtaImport: build.mutation<CtaImportStageDTO, FormData>({
      query: (body) => ({ url: buildCtaUrl('imports/stage'), method: 'POST', body }),
    }),

    prepareCtaImport: build.mutation<CtaImportPreviewDTO, { sessionId: string; body: CtaImportMapping }>({
      query: ({ sessionId, body }) => ({
        url: buildCtaUrl(`imports/i/${sessionId}/prepare`),
        method: 'POST',
        body,
      }),
    }),

    commitCtaImport: build.mutation<CtaImportResultDTO, { sessionId: string; body: CtaImportCommitRequest }>({
      query: ({ sessionId, body }) => ({
        url: buildCtaUrl(`imports/i/${sessionId}/commit`),
        method: 'POST',
        body,
      }),
      invalidatesTags: [{ type: 'CtaTags', id: LIST_TAG }, { type: 'CtaTagSets', id: LIST_TAG }, 'CtaModelConfig'],
    }),

    cancelCtaImport: build.mutation<void, string>({
      query: (sessionId) => ({ url: buildCtaUrl(`imports/i/${sessionId}`), method: 'DELETE' }),
    }),

    downloadCtaSample: build.mutation<Blob, void>({
      query: () => ({
        url: buildCtaUrl('imports/sample'),
        responseHandler: (response) => response.blob(),
      }),
    }),
  }),
});

export const {
  useGetCtaStatusQuery,
  useLazyAutocompleteCtaQuery,
  useListCtaTagsQuery,
  useGetCtaTagQuery,
  useUpdateCtaTagMutation,
  useDeleteCtaTagMutation,
  useBulkMutateCtaTagsMutation,
  useListCtaTagSetsQuery,
  useCreateCtaTagSetMutation,
  useUpdateCtaTagSetMutation,
  useDeleteCtaTagSetMutation,
  useListCtaSyntaxProfilesQuery,
  useCreateCtaSyntaxProfileMutation,
  useUpdateCtaSyntaxProfileMutation,
  useDeleteCtaSyntaxProfileMutation,
  useGetCtaModelConfigQuery,
  useSetCtaModelConfigMutation,
  useStageCtaImportMutation,
  usePrepareCtaImportMutation,
  useCommitCtaImportMutation,
  useCancelCtaImportMutation,
  useDownloadCtaSampleMutation,
} = clipTagAutocompleteApi;

import { beforeEach, describe, expect, it, vi } from 'vitest';

const http = vi.hoisted(() => ({ apiFetch: vi.fn(), apiFetchJson: vi.fn() }));

vi.mock('@platform/transport/http', () => http);

import { commitClipImport, listClipTags, prepareClipImport, searchClipTags } from './api';

const lastCall = (): { body: unknown; method: string | undefined; url: URL } => {
  const [path, init] = http.apiFetchJson.mock.calls.at(-1) as [string, RequestInit | undefined];

  return {
    body: typeof init?.body === 'string' ? JSON.parse(init.body) : init?.body,
    method: init?.method,
    url: new URL(path, 'https://backend.test'),
  };
};

beforeEach(() => {
  http.apiFetch.mockReset().mockResolvedValue(new Response(null, { status: 204 }));
  http.apiFetchJson.mockReset();
});

describe('searching for prompts', () => {
  it('sends the model, category and page as the backend names them, and maps the rendered tags', async () => {
    http.apiFetchJson.mockResolvedValue([
      { canonical_content: 'blue hair', id: '1', popularity: 5, rendered_content: 'blue_hair', tag_type: 'general' },
      { canonical_content: 'x', id: '2', rendered_content: 'x', tag_type: 'artist' },
    ]);

    const candidates = await searchClipTags({
      category: 'character',
      limit: 51,
      modelId: 'model-1',
      offset: 50,
      query: 'blue h',
    });

    const { url } = lastCall();

    expect(url.pathname).toBe('/api/v1/clip_tag_autocomplete/autocomplete');
    expect(Object.fromEntries(url.searchParams)).toEqual({
      limit: '51',
      model_id: 'model-1',
      offset: '50',
      q: 'blue h',
      tag_filter: 'character',
    });
    expect(candidates).toEqual([
      { content: 'blue hair', id: '1', popularity: 5, renderedContent: 'blue_hair', type: 'general' },
      { content: 'x', id: '2', popularity: null, renderedContent: 'x', type: 'artist' },
    ]);
  });
});

describe('the tag list', () => {
  it('filters by search, type and set, or by "no set", and pages by cursor', async () => {
    http.apiFetchJson.mockResolvedValue({
      items: [{ canonical_content: 'a', id: '1', popularity: null, tag_type: 'other' }],
      next_cursor: 'next',
      total_count: 7,
    });

    const page = await listClipTags(
      { search: 'bl', tagSetId: 'set-1', tagType: 'artist' },
      { cursor: 'abc', limit: 50 }
    );

    expect(Object.fromEntries(lastCall().url.searchParams)).toEqual({
      cursor: 'abc',
      limit: '50',
      q: 'bl',
      tag_set_id: 'set-1',
      tag_type: 'artist',
    });
    expect(page).toEqual({
      items: [{ content: 'a', id: '1', popularity: null, type: 'other' }],
      nextCursor: 'next',
      total: 7,
    });

    await listClipTags({ uncategorized: true });

    expect(Object.fromEntries(lastCall().url.searchParams)).toEqual({ uncategorized: 'true' });
  });
});

describe('importing', () => {
  it('sends the column mapping and maps the preview', async () => {
    http.apiFetchJson.mockResolvedValue({
      preview: [{ canonical_content: 'a', tag_type: 'other' }],
      session_id: 's-1',
      summary: {
        invalid_popularity_to_unknown: 1,
        rows_read: 5,
        skipped_rows: 2,
        unknown_types_to_other: 3,
        valid_rows: 4,
      },
    });

    const preview = await prepareClipImport('s-1', {
      delimiter: '\t',
      firstRowContainsColumnNames: true,
      popularityColumn: 2,
      tagColumn: 0,
      typeColumn: null,
    });

    expect(lastCall().url.pathname).toBe('/api/v1/clip_tag_autocomplete/imports/i/s-1/prepare');
    expect(lastCall().body).toEqual({
      delimiter: '\t',
      first_row_contains_column_names: true,
      popularity_column: 2,
      tag_column: 0,
      type_column: null,
    });
    expect(preview).toEqual({
      preview: [{ content: 'a', popularity: null, type: 'other' }],
      sessionId: 's-1',
      summary: { invalidPopularityToUnknown: 1, rowsRead: 5, skippedRows: 2, unknownTypesToOther: 3, validRows: 4 },
    });
  });

  it.each([
    [{ type: 'uncategorized' as const }, { type: 'uncategorized' }],
    [
      { name: 'Set', type: 'new_set' as const },
      { name: 'Set', type: 'new_set' },
    ],
    [
      { mode: 'replace' as const, tagSetId: 'set-1', type: 'existing_set' as const },
      { mode: 'replace', tag_set_id: 'set-1', type: 'existing_set' },
    ],
  ])('commits to a destination as the backend expects it', async (destination, expected) => {
    http.apiFetchJson.mockResolvedValue({
      existing_tags_merged: 1,
      new_tags: 2,
      popularity_updated: 3,
      skipped_rows: 4,
      type_conflicts_ignored: 5,
    });

    const result = await commitClipImport('s-1', destination);

    expect(lastCall().body).toEqual({ destination: expected });
    expect(result).toEqual({
      existingTagsMerged: 1,
      newTags: 2,
      popularityUpdated: 3,
      skippedRows: 4,
      typeConflictsIgnored: 5,
    });
  });
});

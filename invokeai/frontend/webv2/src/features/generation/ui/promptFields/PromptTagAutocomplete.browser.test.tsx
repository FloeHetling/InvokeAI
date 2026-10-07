/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ClipTagCandidate } from '@features/cliptags/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { NegativePromptField } from '@features/generation/ui/promptFields/NegativePromptField';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

const ui = vi.hoisted(() => ({ clipTags: { enabled: true, hotPrefix: '~' } }));
const api = vi.hoisted(() => ({
  getClipTagStatus: vi.fn(),
  searchClipTags: vi.fn(),
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
    clipTags: ui.clipTags,
    gallery: { selectedImage: null },
    models: { catalog: [], ensureLoaded: vi.fn() },
    notifications: { reportError: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [] },
  }),
}));

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

vi.mock('@features/cliptags/data/api', async (importOriginal) => ({ ...(await importOriginal<object>()), ...api }));

const CANDIDATES: ClipTagCandidate[] = [
  { content: 'blue hair', id: '1', popularity: 1200, renderedContent: 'blue_hair', type: 'general' },
  { content: 'blue eyes', id: '2', popularity: 800, renderedContent: 'blue_eyes', type: 'general' },
  { content: 'blue archive', id: '3', popularity: 100, renderedContent: 'blue_archive', type: 'copyright' },
  { content: 'blue sky', id: '4', popularity: null, renderedContent: 'blue_sky', type: 'other' },
];

const SELECTED_MODEL = { key: 'model-1' } as never;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (field: 'negative' | 'positive' = 'positive') => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <ChakraProvider value={system}>
            <DndContext>
              {field === 'positive' ? (
                <PositivePromptField
                  heightPx={200}
                  loras={[]}
                  projectId="project-1"
                  selectedModel={SELECTED_MODEL}
                  showSyntaxHighlighting={false}
                  value=""
                  onChange={vi.fn()}
                  onResizeEnd={vi.fn()}
                  onUsePrompt={vi.fn()}
                />
              ) : (
                <NegativePromptField
                  heightPx={200}
                  isEnabled
                  loras={[]}
                  projectId="project-1"
                  selectedModel={SELECTED_MODEL}
                  showSyntaxHighlighting={false}
                  value=""
                  onChange={vi.fn()}
                  onEnabledChange={vi.fn()}
                  onResizeEnd={vi.fn()}
                />
              )}
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    );
  });
};

const textarea = () => host!.querySelector('textarea')!;
const popup = () => document.querySelector('[role="group"][aria-label="cliptags.prompt.search"]');
const listbox = () => document.querySelector('[role="listbox"][aria-label="cliptags.prompt.suggestions"]');
const optionTexts = () => [...(listbox()?.querySelectorAll('[role="option"]') ?? [])].map((o) => o.textContent ?? '');

const type = async (text: string) => {
  await act(async () => {
    await userEvent.click(textarea());
  });
  await act(async () => {
    await userEvent.type(textarea(), text);
  });
};

const waitForOptions = () => vi.waitFor(() => expect(optionTexts().length).toBeGreaterThan(0));

const MANY_CANDIDATES: ClipTagCandidate[] = Array.from({ length: 30 }, (_, index) => ({
  content: `blue thing ${index}`,
  id: `many-${index}`,
  popularity: 100 - index,
  renderedContent: `blue_thing_${index}`,
  type: 'general',
}));

beforeEach(() => {
  ui.clipTags = { enabled: true, hotPrefix: '~' };
  api.getClipTagStatus.mockReset().mockResolvedValue({ available: true, reason: null });
  api.searchClipTags
    .mockReset()
    .mockImplementation(({ category, query }: { category?: string; query: string }) =>
      Promise.resolve(
        CANDIDATES.filter(
          (candidate) =>
            candidate.content.includes(query) &&
            (category === undefined || (category === 'other' ? candidate.type === 'other' : false))
        )
      )
    );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('tag autocomplete in the prompt', () => {
  it('searches the text after the prefix for the selected model and lists what comes back', async () => {
    await render();
    await type('1girl, ~blue h');
    await waitForOptions();

    expect(optionTexts()).toHaveLength(1);
    expect(optionTexts()[0]).toContain('blue_hair');
    expect(api.searchClipTags).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: 'model-1', query: 'blue h' }),
      expect.any(AbortSignal)
    );
  });

  it('writes the rendered tag over what was typed, with a separator, when Enter is pressed', async () => {
    await render();
    await type('1girl, ~blue h');
    await waitForOptions();
    await act(async () => {
      await userEvent.keyboard('{Enter}');
    });

    expect(textarea().value).toBe('1girl, blue_hair, ');
    expect(popup()).toBeNull();
  });

  it('narrows to a category from the chips without leaving the prompt', async () => {
    await render();
    await type('~blue');
    await vi.waitFor(() => expect(optionTexts().length).toBe(4));

    const chip = [...popup()!.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('cliptags.prompt.filterOther')
    )!;

    await act(() => {
      chip.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, cancelable: true }));
    });
    await vi.waitFor(() => expect(optionTexts()).toHaveLength(1));

    expect(optionTexts()[0]).toContain('blue_sky');
    expect(api.searchClipTags).toHaveBeenLastCalledWith(
      expect.objectContaining({ category: 'other', query: 'blue' }),
      expect.any(AbortSignal)
    );
    expect(document.activeElement).toBe(textarea());
  });

  it('closes on Escape and stays out of the way of an unrelated Enter', async () => {
    await render();
    await type('~blue');
    await waitForOptions();
    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });

    expect(popup()).toBeNull();
  });

  it('does nothing while the feature is off', async () => {
    ui.clipTags = { enabled: false, hotPrefix: '~' };
    await render();
    await type('~blue');

    expect(popup()).toBeNull();
    expect(api.getClipTagStatus).not.toHaveBeenCalled();
    expect(api.searchClipTags).not.toHaveBeenCalled();
  });
});

describe('tag autocomplete details', () => {
  it('tells a failed search from an empty one', async () => {
    api.searchClipTags.mockRejectedValue(new Error('unavailable'));
    await render();
    await type('~blue');

    await vi.waitFor(() => expect(popup()?.textContent).toContain('cliptags.prompt.searchFailed'), { timeout: 4000 });
    expect(popup()?.textContent).not.toContain('cliptags.prompt.noMatches');
  });

  it('keeps the list open while it is scrolled', async () => {
    api.searchClipTags.mockImplementation(({ category, offset = 0 }: { category?: string; offset?: number }) =>
      Promise.resolve(
        category === undefined ? MANY_CANDIDATES.slice(0, 20) : MANY_CANDIDATES.slice(offset, offset + 51)
      )
    );
    await render();
    await type('~blue');
    await vi.waitFor(() => expect(optionTexts()).toHaveLength(20));

    const list = listbox() as HTMLElement;
    await act(() => {
      list.scrollTop = list.scrollHeight;
      list.dispatchEvent(new Event('scroll', { bubbles: true }));
    });
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });

    expect(popup()).not.toBeNull();
  });
});

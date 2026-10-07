import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { expect, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { resetClipTagApi } from './clipTagApi.testing';

/**
 * Browser-test harness for the manager. A test file mocks the transport and translations before importing this:
 *
 *   vi.mock('@features/cliptags/data/api', () => import('./clipTagApi.testing'));
 *   vi.mock('react-i18next', () => import('./clipTagsI18n.testing'));
 */

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | undefined;
let root: Root | undefined;

/** The element the page renders into. */
export const getHost = (): HTMLDivElement => {
  expect(host).toBeDefined();

  return host!;
};

export const setUpClipTagsPage = async (): Promise<void> => {
  await page.viewport(1100, 700);
  applyThemeToRoot(DEFAULT_THEME_ID);
  resetClipTagApi();
};

export const renderClipTagsPage = async (): Promise<void> => {
  host = document.createElement('div');
  host.style.height = '720px';
  host.style.width = '900px';
  document.body.append(host);
  root = createRoot(host);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { ClipTagsPage } = await import('./ClipTagsPage');

  await act(() => {
    root!.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <ClipTagsPage />
        </QueryClientProvider>
      </ChakraProvider>
    );
  });
};

/** Unmounts the page, as a navigation or sign-out would. Safe to call twice. */
export const unmountClipTagsPage = async (): Promise<void> => {
  await act(() => root?.unmount());
  root = undefined;
};

export const tearDownClipTagsPage = async (): Promise<void> => {
  await unmountClipTagsPage();
  host?.remove();
  host = undefined;
};

export const buttonWithText = (text: string, scope: ParentNode = document): HTMLButtonElement => {
  const button = [...scope.querySelectorAll<HTMLButtonElement>('button')].find((candidate) =>
    candidate.textContent?.includes(text)
  );

  expect(button, text).toBeDefined();

  return button!;
};

/** The checkbox input of a labelled list row or header. */
export const checkbox = (label: string): HTMLInputElement => {
  const input = getHost().querySelector<HTMLElement>(`[aria-label="${label}"]`)?.querySelector('input');

  expect(input, label).not.toBeNull();

  return input!;
};

/** The open dialog, or the open dialog whose text includes `text` when several are stacked. */
export const waitForDialog = (text = ''): Promise<HTMLElement> =>
  vi.waitFor(() => {
    const dialog = [
      ...document.querySelectorAll<HTMLElement>(
        '[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"]'
      ),
    ].find((candidate) => candidate.textContent?.includes(text));

    expect(dialog, text).toBeDefined();

    return dialog!;
  });

export const waitForNoDialog = (): Promise<void> =>
  vi.waitFor(() =>
    expect(
      document.querySelector('[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"]')
    ).toBeNull()
  );

export const waitForElement = <T extends HTMLElement>(selector: string, scope: ParentNode = document): Promise<T> =>
  vi.waitFor(() => {
    const element = scope.querySelector<T>(selector);

    expect(element, selector).not.toBeNull();

    return element!;
  });

export const waitForTags = (): Promise<void> => vi.waitFor(() => expect(getHost().textContent).toContain('red hair'));

export const selectTab = async (value: 'syntaxProfiles' | 'tagSets' | 'tags'): Promise<void> => {
  await userEvent.click(await waitForElement(`[role="tab"][data-value="${value}"]`, getHost()));
};

/** Opens the `index`th Select inside `scope` and picks the option with this value. */
export const chooseOption = async (scope: ParentNode, value: string, index = 0): Promise<void> => {
  const trigger = await vi.waitFor(() => {
    const found = scope.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="trigger"]')[index];

    expect(found, `select ${index}`).toBeDefined();

    return found!;
  });

  await userEvent.click(trigger);
  await userEvent.click(await waitForElement(`[role="option"][data-value="${value}"]`));
};

export const inputWithValue = (scope: ParentNode, value: string): Promise<HTMLInputElement> =>
  waitForElement<HTMLInputElement>(`input[value="${value}"]`, scope);

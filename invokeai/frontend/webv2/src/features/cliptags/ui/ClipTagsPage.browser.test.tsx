import { act } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import * as api from './clipTagApi.testing';
import {
  buttonWithText,
  getHost,
  inputWithValue,
  renderClipTagsPage,
  setUpClipTagsPage,
  tearDownClipTagsPage,
  waitForDialog,
  waitForNoDialog,
  waitForTags,
} from './clipTagsHarness.testing';

vi.mock('@features/cliptags/data/api', () => import('./clipTagApi.testing'));
vi.mock('react-i18next', () => import('./clipTagsI18n.testing'));

describe('ClipTagsPage', () => {
  beforeEach(setUpClipTagsPage);
  afterEach(tearDownClipTagsPage);

  it('explains why the manager is unavailable instead of showing its tabs', async () => {
    api.getClipTagStatus.mockResolvedValue({ available: false, reason: 'fts5_unavailable' });
    await renderClipTagsPage();

    await vi.waitFor(() => expect(getHost().textContent).toContain('cliptags.manager.unavailable'));
    // The title and the reason are different sentences.
    expect(getHost().textContent).toContain('cliptags.unavailable.fts5_unavailable');
    expect(getHost().querySelector('[role="tab"]')).toBeNull();
    expect(api.listClipTags).not.toHaveBeenCalled();
  });

  it('saves only the fields that were edited', async () => {
    await renderClipTagsPage();
    await waitForTags();

    await act(() => buttonWithText('red hair', getHost()).click());
    const dialog = await waitForDialog();

    await userEvent.fill(await inputWithValue(dialog, 'red hair'), 'crimson hair');
    await act(() => buttonWithText('common.save', dialog).click());

    await vi.waitFor(() => expect(api.updateClipTag).toHaveBeenCalledWith('t1', { content: 'crimson hair' }));
  });

  it('deletes a tag from its editor after a confirmation that names it', async () => {
    await renderClipTagsPage();
    await waitForTags();

    await act(() => buttonWithText('red hair', getHost()).click());
    const editor = await waitForDialog('cliptags.manager.editor.title');

    await inputWithValue(editor, 'red hair');
    await act(() => buttonWithText('common.delete', editor).click());
    const confirm = await waitForDialog('cliptags.manager.editor.deleteTitle');

    expect(confirm.textContent).toContain('cliptags.manager.editor.deleteBody(name=red hair)');
    expect(api.deleteClipTag).not.toHaveBeenCalled();
    await act(() => buttonWithText('common.delete', confirm).click());

    await vi.waitFor(() => expect(api.deleteClipTag).toHaveBeenCalledWith('t1'));
    // Both dialogs close, and the list is read again.
    await waitForNoDialog();
    await vi.waitFor(() => expect(api.listClipTags.mock.calls.length).toBeGreaterThan(1));
  });
});

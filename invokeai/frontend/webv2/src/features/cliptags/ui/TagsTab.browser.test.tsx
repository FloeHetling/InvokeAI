import { act } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import * as api from './clipTagApi.testing';
import {
  buttonWithText,
  checkbox,
  chooseOption,
  getHost,
  renderClipTagsPage,
  selectTab,
  setUpClipTagsPage,
  tearDownClipTagsPage,
  waitForDialog,
  waitForElement,
  waitForNoDialog,
  waitForTags,
} from './clipTagsHarness.testing';

vi.mock('@features/cliptags/data/api', () => import('./clipTagApi.testing'));
vi.mock('react-i18next', () => import('./clipTagsI18n.testing'));

const SELECT_ALL = 'cliptags.manager.tags.selectAll';
const selectTag = (name: string): string => `cliptags.manager.tags.selectTag(name=${name})`;

const openBulkAction = async (kind: 'add_to_set' | 'delete' | 'remove_from_set' | 'set_type'): Promise<HTMLElement> => {
  await userEvent.click(buttonWithText('cliptags.manager.bulk.actions', getHost()));
  await userEvent.click(await waitForElement(`[role="menuitem"][data-value="${kind}"]`));

  return waitForDialog(`cliptags.manager.bulk.titles.${kind}`);
};

describe('Tags tab bulk operations', () => {
  beforeEach(setUpClipTagsPage);
  afterEach(async () => {
    vi.restoreAllMocks();
    await tearDownClipTagsPage();
  });

  it('deletes every tag matching the filter, sending the filter rather than the loaded ids', async () => {
    await renderClipTagsPage();
    await waitForTags();

    await act(() => checkbox(SELECT_ALL).click());
    await act(() => buttonWithText('cliptags.manager.tags.selectAllMatching', getHost()).click());
    expect(getHost().textContent).toContain('cliptags.manager.tags.selected(count=120');

    const dialog = await openBulkAction('delete');

    expect(dialog.textContent).toContain('cliptags.manager.bulk.deleteWarning');
    await act(() => buttonWithText('cliptags.manager.bulk.apply.delete', dialog).click());

    await vi.waitFor(() =>
      expect(api.bulkMutateClipTags).toHaveBeenCalledWith({ filter: {}, mode: 'filter' }, { type: 'delete' })
    );
    // The write refreshes the list, and the selection does not outlive the rows it described.
    await vi.waitFor(() => expect(api.listClipTags.mock.calls.length).toBeGreaterThan(1));
    await vi.waitFor(() => expect(getHost().textContent).not.toContain('cliptags.manager.tags.selected('));
  });

  it.each([
    { kind: 'add_to_set', operation: { tagSetId: 's1', type: 'add_to_set' }, option: 's1' },
    { kind: 'remove_from_set', operation: { tagSetId: 's1', type: 'remove_from_set' }, option: 's1' },
    { kind: 'set_type', operation: { tagType: 'artist', type: 'set_type' }, option: 'artist' },
  ] as const)('applies "$kind" to the selected tags and clears the selection', async ({ kind, operation, option }) => {
    await renderClipTagsPage();
    await waitForTags();

    await act(() => checkbox(selectTag('red hair')).click());
    await act(() => checkbox(selectTag('blue eyes')).click());
    const dialog = await openBulkAction(kind);

    expect(dialog.textContent).toContain('cliptags.manager.bulk.selected(count=2');

    if (kind !== 'set_type') {
      // A tag set has to be chosen before the operation can run.
      expect(buttonWithText(`cliptags.manager.bulk.apply.${kind}`, dialog).disabled).toBe(true);
    }

    await chooseOption(dialog, option);
    await act(() => buttonWithText(`cliptags.manager.bulk.apply.${kind}`, dialog).click());

    await vi.waitFor(() =>
      expect(api.bulkMutateClipTags).toHaveBeenCalledWith({ ids: ['t1', 't2'], mode: 'ids' }, operation)
    );
    await waitForNoDialog();
    expect(getHost().textContent).not.toContain('cliptags.manager.tags.selected(');
  });

  it('never applies "all matching" to a wider filter after its tag set was deleted on the Tag sets tab', async () => {
    await renderClipTagsPage();
    await waitForTags();

    // Follow the tag set's "View tags" so the list is filtered to it, then select everything that matches.
    await selectTab('tagSets');
    await userEvent.click(await waitForElement('[aria-label^="cliptags.manager.tagSets.viewTagsNamed"]', getHost()));
    await vi.waitFor(() =>
      expect(api.listClipTags).toHaveBeenLastCalledWith(
        expect.objectContaining({ tagSetId: 's1' }),
        expect.anything(),
        expect.anything()
      )
    );
    await waitForTags();
    await act(() => checkbox(SELECT_ALL).click());
    await act(() => buttonWithText('cliptags.manager.tags.selectAllMatching', getHost()).click());
    expect(getHost().textContent).toContain('cliptags.manager.tags.selected(count=120');

    // Deleting the tag set makes the Tags tab's filter fall back to "any set".
    api.deleteClipTagSet.mockResolvedValue(undefined);
    api.listClipTagSets.mockResolvedValue([]);
    await selectTab('tagSets');
    await act(() =>
      getHost().querySelector<HTMLElement>('[aria-label^="cliptags.manager.tagSets.deleteNamed"]')!.click()
    );
    const confirm = await waitForDialog('cliptags.manager.tagSets.deleteTitle');

    await act(() => buttonWithText('common.delete', confirm).click());
    await vi.waitFor(() => expect(api.deleteClipTagSet).toHaveBeenCalledWith('s1'));
    await waitForNoDialog();

    await selectTab('tags');
    await vi.waitFor(() => expect(api.listClipTags).toHaveBeenLastCalledWith({}, expect.anything(), expect.anything()));

    // The selection described the old filter; it is gone, so no bulk action can reach the whole database.
    await vi.waitFor(() => expect(getHost().textContent).not.toContain('cliptags.manager.tags.selected('));
    expect(getHost().textContent).not.toContain('cliptags.manager.tags.allMatchingSelected');
    expect(
      [...getHost().querySelectorAll('button')].some((button) =>
        button.textContent?.includes('cliptags.manager.bulk.actions')
      )
    ).toBe(false);
    expect(api.bulkMutateClipTags).not.toHaveBeenCalled();
  });
});

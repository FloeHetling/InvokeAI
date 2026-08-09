import {
  CtaRequestCoordinator,
  getCtaFilteredPage,
  isCtaQueryEligible,
  normalizeCtaQuery,
} from 'features/clipTagAutocomplete/hooks/useClipTagAutocomplete';
import { afterEach, describe, expect, it, vi } from 'vitest';

describe('CTA query normalization', () => {
  it('treats underscores and physical spaces as equivalent word separators', () => {
    expect(normalizeCtaQuery('  red__hair  ')).toBe('red hair');
    expect(normalizeCtaQuery('red   hair')).toBe('red hair');
  });

  it('does not become request-eligible until two normalized characters are present', () => {
    expect(isCtaQueryEligible('', true)).toBe(false);
    expect(isCtaQueryEligible('a', true)).toBe(false);
    expect(isCtaQueryEligible('ab', true)).toBe(true);
    expect(isCtaQueryEligible('ab', false)).toBe(false);
  });
});

describe('CTA filtered pagination', () => {
  it('does not advertise another page when exactly 50 results exist', () => {
    const data = Array.from({ length: 50 }, (_, index) => index);
    const page = getCtaFilteredPage(data);

    expect(page.items).toEqual(data);
    expect(page.hasMore).toBe(false);
  });

  it('uses the 51st result only as a lookahead signal', () => {
    const data = Array.from({ length: 51 }, (_, index) => index);
    const page = getCtaFilteredPage(data);

    expect(page.items).toEqual(data.slice(0, 50));
    expect(page.hasMore).toBe(true);
  });
});

describe('CTA request lifecycle', () => {
  afterEach(() => vi.useRealTimers());

  it('waits for the debounce and does not repeat an unchanged query', async () => {
    vi.useFakeTimers();
    const trigger = vi.fn(() => ({ abort: vi.fn(), unsubscribe: vi.fn(), unwrap: () => Promise.resolve(['result']) }));
    const coordinator = new CtaRequestCoordinator<string[]>();
    const onSuccess = vi.fn();
    const callbacks = { onStart: vi.fn(), onSuccess, onFailure: vi.fn() };

    expect(coordinator.schedule('ab:', { q: 'ab' }, trigger, callbacks)).toBe(true);
    const replacementTrigger = vi.fn(() => ({
      abort: vi.fn(),
      unsubscribe: vi.fn(),
      unwrap: () => Promise.resolve(['other']),
    }));
    expect(coordinator.schedule('ab:', { q: 'ab' }, replacementTrigger, callbacks)).toBe(false);

    await vi.advanceTimersByTimeAsync(199);
    expect(trigger).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1);
    expect(trigger).toHaveBeenCalledOnce();
    expect(onSuccess).toHaveBeenCalledWith(['result']);

    await vi.advanceTimersByTimeAsync(5_000);
    expect(trigger).toHaveBeenCalledOnce();
  });

  it('aborts an obsolete in-flight request and ignores its stale result', async () => {
    vi.useFakeTimers();
    let resolveFirst: (value: string[]) => void = () => undefined;
    const firstPromise = new Promise<string[]>((resolve) => {
      resolveFirst = resolve;
    });
    const firstAbort = vi.fn();
    const trigger = vi
      .fn()
      .mockReturnValueOnce({ abort: firstAbort, unsubscribe: vi.fn(), unwrap: () => firstPromise })
      .mockReturnValueOnce({ abort: vi.fn(), unsubscribe: vi.fn(), unwrap: () => Promise.resolve(['current']) });
    const coordinator = new CtaRequestCoordinator<string[]>();
    const onSuccess = vi.fn();
    const callbacks = { onStart: vi.fn(), onSuccess, onFailure: vi.fn() };

    coordinator.schedule('ab:', { q: 'ab' }, trigger, callbacks);
    await vi.advanceTimersByTimeAsync(200);

    coordinator.schedule('abc:', { q: 'abc' }, trigger, callbacks);
    expect(firstAbort).toHaveBeenCalledOnce();

    resolveFirst(['stale']);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(200);

    expect(onSuccess).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledWith(['current']);
  });

  it('cancels a pending debounce without issuing a request', async () => {
    vi.useFakeTimers();
    const trigger = vi.fn(() => ({ abort: vi.fn(), unsubscribe: vi.fn(), unwrap: () => Promise.resolve([]) }));
    const coordinator = new CtaRequestCoordinator<string[]>();

    coordinator.schedule('ab:', { q: 'ab' }, trigger, { onStart: vi.fn(), onSuccess: vi.fn(), onFailure: vi.fn() });
    coordinator.cancel();
    await vi.advanceTimersByTimeAsync(1_000);

    expect(trigger).not.toHaveBeenCalled();
  });

  it('does not retry a failed autocomplete request while idle', async () => {
    vi.useFakeTimers();
    const trigger = vi.fn(() => ({
      abort: vi.fn(),
      unsubscribe: vi.fn(),
      unwrap: () => Promise.reject(new Error('network failure')),
    }));
    const onFailure = vi.fn();
    const coordinator = new CtaRequestCoordinator<string[]>();

    coordinator.schedule('ab:', { q: 'ab' }, trigger, { onStart: vi.fn(), onSuccess: vi.fn(), onFailure });
    await vi.advanceTimersByTimeAsync(200);
    await Promise.resolve();

    expect(trigger).toHaveBeenCalledOnce();
    expect(onFailure).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(30_000);
    expect(trigger).toHaveBeenCalledOnce();
  });

  it('starts filtered pagination immediately and refuses overlapping pages', async () => {
    vi.useFakeTimers();
    let resolvePage: (value: string[]) => void = () => undefined;
    const pagePromise = new Promise<string[]>((resolve) => {
      resolvePage = resolve;
    });
    const trigger = vi
      .fn()
      .mockReturnValueOnce({ abort: vi.fn(), unsubscribe: vi.fn(), unwrap: () => Promise.resolve(['first']) })
      .mockReturnValueOnce({ abort: vi.fn(), unsubscribe: vi.fn(), unwrap: () => pagePromise });
    const coordinator = new CtaRequestCoordinator<string[]>();

    coordinator.schedule('ab::other', { q: 'ab', tag_filter: 'other', offset: 0 }, trigger, {
      onStart: vi.fn(),
      onSuccess: vi.fn(),
      onFailure: vi.fn(),
    });
    await vi.advanceTimersByTimeAsync(200);

    const onSuccess = vi.fn();
    const callbacks = { onStart: vi.fn(), onSuccess, onFailure: vi.fn() };
    expect(coordinator.requestMore('ab::other', { q: 'ab', tag_filter: 'other', offset: 1 }, trigger, callbacks)).toBe(
      true
    );
    expect(coordinator.requestMore('ab::other', { q: 'ab', tag_filter: 'other', offset: 1 }, trigger, callbacks)).toBe(
      false
    );
    expect(trigger).toHaveBeenCalledTimes(2);

    resolvePage(['second']);
    await Promise.resolve();
    await Promise.resolve();

    expect(onSuccess).toHaveBeenCalledWith(['second']);
  });
});

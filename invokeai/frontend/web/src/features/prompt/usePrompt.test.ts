import { computeCtaInsertion } from 'features/prompt/usePrompt';
import { describe, expect, it } from 'vitest';

describe('CTA prompt insertion', () => {
  it('supports sequential insertions by carrying the returned caret forward', () => {
    const initialPrompt = 'portrait, background';
    const initialCaret = 'portrait, '.length;

    const first = computeCtaInsertion(initialPrompt, initialCaret, 'fluffy body');
    const second = computeCtaInsertion(first.value, first.caret, 'fluffy tail');

    expect(first.value).toBe('portrait, fluffy body, background');
    expect(second.value).toBe('portrait, fluffy body, fluffy tail, background');
    expect(second.caret).toBe('portrait, fluffy body, fluffy tail, '.length);
  });
});

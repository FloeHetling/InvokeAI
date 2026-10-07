import { describe, expect, it } from 'vitest';

import { getActiveClipTagQuery, getClipTagInsertion, isClipTagQueryEligible, normalizeClipTagQuery } from './prompt';

describe('normalizeClipTagQuery', () => {
  it('treats underscores like spaces and collapses whitespace', () => {
    expect(normalizeClipTagQuery('  blue__hair \t long ')).toBe('blue hair long');
  });

  it('needs two visible characters before searching', () => {
    expect(isClipTagQueryEligible('a')).toBe(false);
    expect(isClipTagQueryEligible(' _ a ')).toBe(false);
    expect(isClipTagQueryEligible('ab')).toBe(true);
  });
});

describe('getActiveClipTagQuery', () => {
  it('finds the text typed after a prefix that starts the prompt', () => {
    expect(getActiveClipTagQuery('~blue ha', 8, '~')).toEqual({ query: 'blue ha', range: { end: 8, start: 0 } });
  });

  it('finds a prefix after a comma and a space, using only the text before the caret', () => {
    const value = '1girl, ~red hair, smile';

    expect(getActiveClipTagQuery(value, 14, '~')).toEqual({ query: 'red ha', range: { end: 14, start: 7 } });
  });

  it('keeps a prefix inside a word literal', () => {
    expect(getActiveClipTagQuery('a~bc', 4, '~')).toBeNull();
    expect(getActiveClipTagQuery('a_~bc', 5, '~')).toBeNull();
  });

  it('opens after punctuation that cannot belong to a word', () => {
    expect(getActiveClipTagQuery('(~bl', 4, '~')).toEqual({ query: 'bl', range: { end: 4, start: 1 } });
  });

  it('ends at a comma or a line break', () => {
    expect(getActiveClipTagQuery('~red, hair', 10, '~')).toBeNull();
    expect(getActiveClipTagQuery('~red\nhair', 9, '~')).toBeNull();
  });

  it('only reacts to the configured prefix', () => {
    expect(getActiveClipTagQuery('*blue', 5, '~')).toBeNull();
    expect(getActiveClipTagQuery('*blue', 5, '*')).toEqual({ query: 'blue', range: { end: 5, start: 0 } });
  });

  it('is inactive without a prefix, and a very long line does not keep a distant prefix open', () => {
    expect(getActiveClipTagQuery('blue', 4, '~')).toBeNull();
    expect(getActiveClipTagQuery(`~${'a'.repeat(200)}`, 201, '~')).toBeNull();
  });
});

describe('getClipTagInsertion', () => {
  it('writes the tag followed by a separator', () => {
    expect(getClipTagInsertion('~blue', { end: 5, start: 0 }, 'blue_hair')).toEqual({
      range: { end: 5, start: 0 },
      text: 'blue_hair, ',
    });
  });

  it('takes over a separator already after the caret', () => {
    const value = '~blue ,  smile';

    expect(getClipTagInsertion(value, { end: 5, start: 0 }, 'blue_hair')).toEqual({
      range: { end: 5 + ' ,  '.length, start: 0 },
      text: 'blue_hair, ',
    });
  });
});
